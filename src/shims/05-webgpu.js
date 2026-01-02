// 05-webgpu.js - WebGPU shims

const gfx = globalThis.__gfx;
if (!gfx) {
  console.log("[shims] WebGPU skipped (no native ops)");
} else {
  // Enums
  globalThis.GPUBufferUsage = {
    MAP_READ: 1,
    MAP_WRITE: 2,
    COPY_SRC: 4,
    COPY_DST: 8,
    INDEX: 16,
    VERTEX: 32,
    UNIFORM: 64,
    STORAGE: 128,
    INDIRECT: 256,
    QUERY_RESOLVE: 512,
  };
  globalThis.GPUTextureUsage = {
    COPY_SRC: 1,
    COPY_DST: 2,
    TEXTURE_BINDING: 4,
    STORAGE_BINDING: 8,
    RENDER_ATTACHMENT: 16,
  };
  globalThis.GPUMapMode = { READ: 1, WRITE: 2 };

  // ImageBitmap
  class ImageBitmap {
    constructor(w, h, rid) {
      this._width = w;
      this._height = h;
      this._rid = rid;
      this._closed = false;
    }
    get width() {
      return this._width;
    }
    get height() {
      return this._height;
    }
    close() {
      if (!this._closed && this._rid != null) {
        gfx.op_gfx_decoded_image_drop(this._rid);
        this._rid = null;
      }
      this._closed = true;
    }
  }

  globalThis.ImageBitmap = ImageBitmap;

  globalThis.createImageBitmap = async function (
    source,
    optionsOrSx,
    sy,
    sw,
    sh,
    options
  ) {
    const opts =
      typeof optionsOrSx === "object" && sy === undefined
        ? optionsOrSx
        : options || {};

    if (!(source instanceof Blob))
      throw new Error("Only Blob sources supported");

    const bytes = new Uint8Array(await source.arrayBuffer());

    const format =
      {
        "image/png": "png",
        "image/jpeg": "jpeg",
        "image/jpg": "jpeg",
        "image/webp": "webp",
      }[source.type] ||
      sniffFormat(bytes) ||
      "auto";

    const { rid, width, height } = gfx.op_gfx_decode_image_store(bytes, {
      format,
      resize_width: opts.resizeWidth ?? null,
      resize_height: opts.resizeHeight ?? null,
      resize_quality: opts.resizeQuality ?? null,
      image_orientation: opts.imageOrientation ?? null,
    });

    return new ImageBitmap(width, height, rid);
  };

  function sniffFormat(b) {
    if (b[0] === 0x89 && b[1] === 0x50 && b[2] === 0x4e && b[3] === 0x47)
      return "png";
    if (b[0] === 0xff && b[1] === 0xd8 && b[2] === 0xff) return "jpeg";
    if (b[0] === 0x52 && b[1] === 0x49 && b[8] === 0x57 && b[9] === 0x45)
      return "webp";
    return null;
  }

  // Canvas context
  class GPUCanvasContext {
    constructor() {
      this.config = null;
      this._pendingDrawCalls = [];
      this._frameSubmitted = false;
    }

    configure(config) {
      this.config = { ...config };
      gfx.op_gfx_surface_configure({
        format: config.format,
        width: globalThis.__windowInnerWidth ?? 800,
        height: globalThis.__windowInnerHeight ?? 600,
        alphaMode: config.alphaMode ?? "opaque",
      });
    }

    getCurrentTexture() {
      this._frameSubmitted = false;
      return { createView: () => ({ __surfaceView: true }) };
    }

    __queueDrawCalls(calls) {
      this._pendingDrawCalls.push(...calls);
    }

    __submitFrame() {
      if (this._frameSubmitted || globalThis.__xrMode) {
        this._pendingDrawCalls = [];
        this._frameSubmitted = true;
        return;
      }
      if (this._pendingDrawCalls.length) {
        gfx.op_gfx_surface_draw({ draw_calls: this._pendingDrawCalls });
        this._pendingDrawCalls = [];
        this._frameSubmitted = true;
      }
    }
  }

  globalThis.GPUCanvasContext = GPUCanvasContext;

  // Emulated buffer
  class EmulatedBuffer {
    constructor(size, usage) {
      this.byteLength = size >>> 0;
      this.usage = usage >>> 0;
      this._mapped = new ArrayBuffer(this.byteLength);
      this.__rid = null;
    }
    getMappedRange() {
      if (!this._mapped) throw new Error("Buffer not mapped");
      return this._mapped;
    }
    unmap() {
      if (!this._mapped) return;
      const u8 = new Uint8Array(this._mapped);
      this.__rid = gfx.op_gfx_device_create_buffer_init(
        this.usage >>> 0,
        null,
        u8
      ).rid;
      this._mapped = null;
    }
  }

  // Emulated texture
  class EmulatedTexture {
    constructor(desc, rid) {
      this.descriptor = desc;
      this.__rid = rid;
      this.width = desc.size?.width ?? desc.size?.[0] ?? 1;
      this.height = desc.size?.height ?? desc.size?.[1] ?? 1;
      this.depthOrArrayLayers =
        desc.size?.depthOrArrayLayers ?? desc.size?.[2] ?? 1;
      this.format = desc.format;
      this.mipLevelCount = desc.mipLevelCount ?? 1;
    }

    createView(viewDesc = {}) {
      const v = gfx.op_gfx_texture_create_view({
        texture_rid: this.__rid,
        label: viewDesc.label ?? null,
        format: viewDesc.format ?? null,
        dimension: viewDesc.dimension ?? null,
        base_mip_level: viewDesc.baseMipLevel ?? null,
        mip_level_count: viewDesc.mipLevelCount ?? null,
        base_array_layer: viewDesc.baseArrayLayer ?? null,
        array_layer_count: viewDesc.arrayLayerCount ?? null,
      });
      return { __rid: v.rid };
    }

    destroy() {}
  }

  // Render pass
  class EmulatedRenderPass {
    constructor(context, desc) {
      this._context = context;
      this._pipelineRid = null;
      this._vertexBuffers = new Map();
      this._indexBuffer = null;
      this._indexFormat = null;
      this._bindGroups = new Map();
      this._drawCalls = [];

      const ca = desc?.colorAttachments?.[0];
      this._colorTargetView = ca?.view;
      this._resolveTarget = ca?.resolveTarget ?? null;

      const hasColor = ca?.view != null;
      const hasResolve = ca?.resolveTarget != null;
      const viewHasRid = ca?.view?.__rid != null;

      this._isRenderToTexture = viewHasRid && !hasResolve;
      this._isDepthOnlyPass =
        !hasColor && desc?.depthStencilAttachment?.view?.__rid != null;

      this._clearColor = null;
      if (ca?.loadOp === "clear" && ca.clearValue) {
        const cv = ca.clearValue;
        this._clearColor = Array.isArray(cv)
          ? { r: cv[0] ?? 0, g: cv[1] ?? 0, b: cv[2] ?? 0, a: cv[3] ?? 1 }
          : { r: cv.r ?? 0, g: cv.g ?? 0, b: cv.b ?? 0, a: cv.a ?? 1 };
      }

      const da = desc?.depthStencilAttachment;
      this._depthAttachment =
        da?.view?.__rid != null
          ? {
              viewRid: da.view.__rid,
              depthClearValue: da.depthClearValue ?? 1.0,
              depthLoadOp: da.depthLoadOp || "clear",
              depthStoreOp: da.depthStoreOp || "store",
              depthReadOnly: !!da.depthReadOnly,
            }
          : null;
    }

    multiDrawIndexedIndirect(indirectBuffer, indirectOffset, drawCount) {
      if (this._pipelineRid == null) return;
      if (!indirectBuffer?.__rid)
        throw new Error("multiDrawIndexedIndirect: no indirect buffer");
      if (!this._indexBuffer?.__rid)
        throw new Error("multiDrawIndexedIndirect: no index buffer set");

      const vRids = [];
      for (const [slot, rid] of this._vertexBuffers) vRids[slot] = rid;

      const bgRids = [];
      for (const [idx, rid] of this._bindGroups) bgRids[idx] = rid;

      const isFirst = this._drawCalls.length === 0;

      this._drawCalls.push({
        pipeline_rid: this._pipelineRid,
        vertex_buffer_rids: vRids,
        bind_group_rids: bgRids,
        clear_color: isFirst ? this._clearColor : null,
        vertex_count: 0,
        instance_count: 0,
        first_vertex: 0,
        first_instance: 0,
        index_buffer_rid: this._indexBuffer.__rid,
        index_format: this._indexFormat || "uint32",
        index_count: 0,
        first_index: 0,
        base_vertex: 0,
        depth_view_rid: isFirst ? this._depthAttachment?.viewRid ?? null : null,
        depth_clear_value: isFirst
          ? this._depthAttachment?.depthClearValue ?? null
          : null,
        depth_load_op: isFirst
          ? this._depthAttachment?.depthLoadOp ?? null
          : null,
        depth_store_op: isFirst
          ? this._depthAttachment?.depthStoreOp ?? null
          : null,
        depth_read_only: isFirst
          ? this._depthAttachment?.depthReadOnly ?? null
          : null,
        // Multi-draw specific
        is_multi_draw: true,
        indirect_buffer_rid: indirectBuffer.__rid,
        indirect_offset: indirectOffset >>> 0,
        draw_count: drawCount >>> 0,
      });
    }

    setPipeline(p) {
      this._pipelineRid = p.__rid;
    }

    setVertexBuffer(slot, buf) {
      if (!buf) throw new Error("setVertexBuffer: null buffer");
      if (buf.__rid == null && buf.unmap) buf.unmap();
      if (buf.__rid == null) throw new Error("setVertexBuffer: no __rid");
      this._vertexBuffers.set(slot >>> 0, buf.__rid);
    }

    setIndexBuffer(buffer, format) {
      if (!buffer) throw new Error("setIndexBuffer: null buffer");
      if (buffer.__rid == null && buffer.unmap) buffer.unmap();
      if (buffer.__rid == null) throw new Error("setIndexBuffer: no __rid");
      this._indexBuffer = buffer;
      this._indexFormat = format || "uint32";
    }

    setBindGroup(index, bg) {
      if (bg?.__rid != null) this._bindGroups.set(index >>> 0, bg.__rid);
    }

    setViewport() {}
    setScissorRect() {}
    setBlendConstant() {}
    setStencilReference() {}
    beginOcclusionQuery() {}
    endOcclusionQuery() {}

    _recordDraw(params) {
      if (this._pipelineRid == null) return;

      const vRids = [];
      for (const [slot, rid] of this._vertexBuffers) vRids[slot] = rid;

      const bgRids = [];
      for (const [idx, rid] of this._bindGroups) bgRids[idx] = rid;

      let indexRid = null,
        indexFormat = null;
      if (params.indexed) {
        if (!this._indexBuffer?.__rid)
          throw new Error("Indexed draw without index buffer");
        indexRid = this._indexBuffer.__rid;
        indexFormat = this._indexFormat || "uint32";
      }

      const isFirst = this._drawCalls.length === 0;

      this._drawCalls.push({
        pipeline_rid: this._pipelineRid,
        vertex_buffer_rids: vRids,
        bind_group_rids: bgRids,
        clear_color: isFirst ? this._clearColor : null,
        vertex_count: params.indexed ? 0 : params.vertexCount ?? 0,
        instance_count: params.instanceCount ?? 1,
        first_vertex: params.indexed ? 0 : params.firstVertex ?? 0,
        first_instance: params.firstInstance ?? 0,
        index_buffer_rid: params.indexed ? indexRid : null,
        index_format: params.indexed ? indexFormat : null,
        index_count: params.indexed ? params.indexCount ?? 0 : 0,
        first_index: params.indexed ? params.firstIndex ?? 0 : 0,
        base_vertex: params.indexed ? params.baseVertex ?? 0 : 0,
        depth_view_rid: isFirst ? this._depthAttachment?.viewRid ?? null : null,
        depth_clear_value: isFirst
          ? this._depthAttachment?.depthClearValue ?? null
          : null,
        depth_load_op: isFirst
          ? this._depthAttachment?.depthLoadOp ?? null
          : null,
        depth_store_op: isFirst
          ? this._depthAttachment?.depthStoreOp ?? null
          : null,
        depth_read_only: isFirst
          ? this._depthAttachment?.depthReadOnly ?? null
          : null,
      });
    }

    draw(vertexCount, instanceCount = 1, firstVertex = 0, firstInstance = 0) {
      this._recordDraw({
        indexed: false,
        vertexCount,
        instanceCount,
        firstVertex,
        firstInstance,
      });
    }

    drawIndexed(
      indexCount,
      instanceCount = 1,
      firstIndex = 0,
      baseVertex = 0,
      firstInstance = 0
    ) {
      this._recordDraw({
        indexed: true,
        indexCount,
        instanceCount,
        firstIndex,
        baseVertex,
        firstInstance,
      });
    }

    drawIndirect() {}

    drawIndexedIndirect(indirectBuffer, indirectOffset) {
      if (this._pipelineRid == null) return;
      if (!indirectBuffer?.__rid)
        throw new Error("drawIndexedIndirect: no indirect buffer");
      if (!this._indexBuffer?.__rid)
        throw new Error("drawIndexedIndirect: no index buffer set");

      const vRids = [];
      for (const [slot, rid] of this._vertexBuffers) vRids[slot] = rid;

      const bgRids = [];
      for (const [idx, rid] of this._bindGroups) bgRids[idx] = rid;

      const isFirst = this._drawCalls.length === 0;

      this._drawCalls.push({
        pipeline_rid: this._pipelineRid,
        vertex_buffer_rids: vRids,
        bind_group_rids: bgRids,
        clear_color: isFirst ? this._clearColor : null,
        vertex_count: 0,
        instance_count: 0,
        first_vertex: 0,
        first_instance: 0,
        index_buffer_rid: this._indexBuffer.__rid,
        index_format: this._indexFormat || "uint32",
        index_count: 0,
        first_index: 0,
        base_vertex: 0,
        depth_view_rid: isFirst ? this._depthAttachment?.viewRid ?? null : null,
        depth_clear_value: isFirst
          ? this._depthAttachment?.depthClearValue ?? null
          : null,
        depth_load_op: isFirst
          ? this._depthAttachment?.depthLoadOp ?? null
          : null,
        depth_store_op: isFirst
          ? this._depthAttachment?.depthStoreOp ?? null
          : null,
        depth_read_only: isFirst
          ? this._depthAttachment?.depthReadOnly ?? null
          : null,
        is_indirect: true,
        indirect_buffer_rid: indirectBuffer.__rid,
        indirect_offset: indirectOffset >>> 0,
      });
    }

    executeBundles(bundles) {
      for (const bundle of bundles) {
        if (!bundle?.__isRenderBundle) continue;
        for (const cmd of bundle.commands) {
          if (cmd.type === "draw") {
            this._pipelineRid = cmd.pipeline?.__rid;
            this._vertexBuffers.clear();
            for (const [slot, info] of cmd.vertexBuffers) {
              if (info.buffer?.__rid != null)
                this._vertexBuffers.set(slot, info.buffer.__rid);
            }
            this._bindGroups.clear();
            for (const [idx, bg] of cmd.bindGroups) {
              if (bg?.__rid != null) this._bindGroups.set(idx, bg.__rid);
            }
            this.draw(
              cmd.vertexCount,
              cmd.instanceCount,
              cmd.firstVertex,
              cmd.firstInstance
            );
          } else if (cmd.type === "drawIndexed") {
            this._pipelineRid = cmd.pipeline?.__rid;
            this._vertexBuffers.clear();
            for (const [slot, info] of cmd.vertexBuffers) {
              if (info.buffer?.__rid != null)
                this._vertexBuffers.set(slot, info.buffer.__rid);
            }
            this._bindGroups.clear();
            for (const [idx, bg] of cmd.bindGroups) {
              if (bg?.__rid != null) this._bindGroups.set(idx, bg.__rid);
            }
            if (cmd.indexBuffer) {
              this._indexBuffer = cmd.indexBuffer;
              this._indexFormat = cmd.indexFormat || "uint32";
            }
            this.drawIndexed(
              cmd.indexCount,
              cmd.instanceCount,
              cmd.firstIndex,
              cmd.baseVertex,
              cmd.firstInstance
            );
          }
        }
      }
    }

    end() {
      if (!this._drawCalls.length) return;

      if (this._isDepthOnlyPass && this._depthAttachment?.viewRid != null) {
        gfx.op_gfx_render_depth_only({
          depth_view_rid: this._depthAttachment.viewRid,
          draw_calls: this._drawCalls,
        });
        return;
      }

      const resolve = this._resolveTarget;
      if (resolve?._isXRSwapchain && resolve.__rid != null) {
        const msaaViewRid = this._colorTargetView?.__rid;
        if (msaaViewRid != null) {
          gfx.op_gfx_render_xr_frame({
            msaa_view_rid: msaaViewRid,
            resolve_view_rid: resolve.__rid,
            draw_calls: this._drawCalls,
            depth_view_rid: this._depthAttachment?.viewRid ?? null,
            depth_clear_value: this._depthAttachment?.depthClearValue ?? 1.0,
            clear_color: this._clearColor,
          });
        }
        return;
      }

      if (this._isRenderToTexture && this._colorTargetView?.__rid != null) {
        gfx.op_gfx_render_to_texture({
          target_view_rid: this._colorTargetView.__rid,
          draw_calls: this._drawCalls,
        });
        return;
      }

      this._context?.__queueDrawCalls?.(this._drawCalls);
    }
  }

  // Command encoder
  class EmulatedEncoder {
    constructor(context) {
      this._context = context;
    }

    beginRenderPass(desc) {
      return new EmulatedRenderPass(this._context, desc);
    }

    copyBufferToBuffer() {}

    copyTextureToTexture(src, dst, copySize) {
      if (!src.texture?.__rid || !dst.texture?.__rid) {
        throw new Error("copyTextureToTexture: texture missing __rid");
      }
      const so = src.origin || {};
      const d = dst.origin || {};
      gfx.op_gfx_copy_texture_to_texture({
        src_texture_rid: src.texture.__rid,
        src_mip_level: src.mipLevel ?? 0,
        src_origin_x: so.x ?? 0,
        src_origin_y: so.y ?? 0,
        src_origin_z: so.z ?? 0,
        dst_texture_rid: dst.texture.__rid,
        dst_mip_level: dst.mipLevel ?? 0,
        dst_origin_x: d.x ?? 0,
        dst_origin_y: d.y ?? 0,
        dst_origin_z: d.z ?? 0,
        width: copySize?.width ?? copySize?.[0] ?? 1,
        height: copySize?.height ?? copySize?.[1] ?? 1,
        depth_or_array_layers:
          copySize?.depthOrArrayLayers ?? copySize?.[2] ?? 1,
      });
    }

    copyTextureToBuffer() {}

    finish() {
      globalThis.__gpuCanvasContext?.__submitFrame();
      return { __finished: true };
    }
  }

  // Feature set helper
  function makeFeaturesSet(arr) {
    const set = new Set(arr || []);
    return {
      has: (n) => set.has(n),
      [Symbol.iterator]: () => set[Symbol.iterator](),
    };
  }

  // GPU adapter/device
  const gpuShim = {
    async requestAdapter() {
      if (globalThis.__xr && !globalThis.__xrSessionInfo) {
        try {
          globalThis.__xrSessionInfo =
            await globalThis.__xr.op_xr_request_session();
        } catch (e) {
          console.log("Native XR not available:", e.message);
        }
      }

      const features = makeFeaturesSet([]);
      const limits = {
        maxTextureDimension1D: 8192,
        maxTextureDimension2D: 8192,
        maxTextureDimension3D: 2048,
        maxBindGroups: 4,
      };

      return {
        features,
        limits,
        async requestDevice(descriptor = {}) {
          const requiredFeatures = descriptor.requiredFeatures || [];
          const requiredLimits = descriptor.requiredLimits || {};
          const actualFeatures = makeFeaturesSet(
            requiredFeatures.filter((f) => features.has(f))
          );
          const deviceLimits = { ...limits, ...requiredLimits };

          const device = {
            features: actualFeatures,
            limits: deviceLimits,
            lost: new Promise(() => {}),

            createShaderModule(desc) {
              return {
                __rid: gfx.op_gfx_device_create_shader({
                  code: desc.code,
                  label: desc.label ?? null,
                }).rid,
              };
            },

            createRenderPipeline(desc) {
              const vertexBuffers = (desc.vertex.buffers || []).map((buf) => ({
                array_stride: buf.arrayStride >>> 0,
                step_mode: buf.stepMode ?? null,
                attributes: (buf.attributes || []).map((a) => ({
                  format: a.format,
                  offset: a.offset >>> 0,
                  shader_location: a.shaderLocation >>> 0,
                })),
              }));

              const prim = desc.primitive || {};
              const depth = desc.depthStencil || null;

              const rid = gfx.op_gfx_device_create_pipeline({
                vertex_module_rid: desc.vertex.module.__rid,
                vertex_entry: desc.vertex.entryPoint ?? "vs_main",
                fragment_module_rid: desc.fragment?.module?.__rid ?? null,
                fragment_entry: desc.fragment?.entryPoint ?? null,
                vertex_buffers: vertexBuffers,
                pipeline_layout_rid:
                  desc.layout === "auto" ? null : desc.layout?.__rid ?? null,
                primitive_topology: prim.topology || null,
                primitive_front_face: prim.frontFace || null,
                primitive_cull_mode: prim.cullMode || null,
                depth_format: depth?.format ?? null,
                depth_write_enabled: depth?.depthWriteEnabled ?? null,
                depth_compare: depth?.depthCompare ?? null,
                color_format: desc.fragment?.targets?.[0]?.format ?? null,
                sample_count: desc.multisample?.count ?? 1,
                alpha_to_coverage_enabled:
                  desc.multisample?.alphaToCoverageEnabled ?? false,
              }).rid;

              return {
                __rid: rid,
                getBindGroupLayout(index) {
                  return {
                    __rid: gfx.op_gfx_pipeline_get_bind_group_layout({
                      pipeline_rid: rid,
                      index: index >>> 0,
                    }).rid,
                  };
                },
              };
            },

            createBuffer(options) {
              const size = options.size >>> 0;
              const usage = options.usage >>> 0;
              if (options.mappedAtCreation)
                return new EmulatedBuffer(size, usage);
              return {
                __rid: gfx.op_gfx_device_create_buffer({
                  size,
                  usage,
                  label: options.label ?? null,
                }).rid,
                byteLength: size,
                usage,
              };
            },

            createTexture(desc) {
              const size = desc.size || {};
              const tDesc = {
                label: desc.label ?? null,
                size: {
                  width: (size.width ?? size[0] ?? 1) >>> 0,
                  height: (size.height ?? size[1] ?? 1) >>> 0,
                  depth_or_array_layers:
                    (size.depthOrArrayLayers ?? size[2] ?? 1) >>> 0,
                },
                mip_level_count: (desc.mipLevelCount ?? 1) >>> 0,
                sample_count: (desc.sampleCount ?? 1) >>> 0,
                dimension: desc.dimension || "2d",
                format: desc.format,
                usage:
                  (desc.usage |
                    GPUTextureUsage.COPY_SRC |
                    GPUTextureUsage.COPY_DST) >>>
                  0,
              };
              return new EmulatedTexture(
                desc,
                gfx.op_gfx_device_create_texture(tDesc).rid
              );
            },

            createSampler(desc = {}) {
              return {
                __rid: gfx.op_gfx_device_create_sampler({
                  label: desc.label ?? null,
                  mag_filter: desc.magFilter || null,
                  min_filter: desc.minFilter || null,
                  mipmap_filter: desc.mipmapFilter || null,
                  address_mode_u: desc.addressModeU || null,
                  address_mode_v: desc.addressModeV || null,
                  address_mode_w: desc.addressModeW || null,
                  compare: desc.compare || null,
                }).rid,
                __isSampler: true,
              };
            },

            createBindGroupLayout(desc) {
              const entries = (desc.entries || []).map((e) => ({
                binding: e.binding >>> 0,
                visibility: e.visibility >>> 0,
                buffer: e.buffer
                  ? {
                      type: e.buffer.type || null,
                      hasDynamicOffset: !!e.buffer.hasDynamicOffset,
                      minBindingSize: e.buffer.minBindingSize ?? null,
                    }
                  : null,
                sampler: e.sampler ? { type: e.sampler.type || null } : null,
                texture: e.texture
                  ? {
                      sampleType: e.texture.sampleType || null,
                      viewDimension: e.texture.viewDimension || null,
                      multisampled: !!e.texture.multisampled,
                    }
                  : null,
                storageTexture: e.storageTexture
                  ? {
                      access: e.storageTexture.access || null,
                      format: e.storageTexture.format || null,
                      viewDimension: e.storageTexture.viewDimension || null,
                    }
                  : null,
              }));
              return {
                __rid: gfx.op_gfx_device_create_bind_group_layout({
                  label: desc.label ?? null,
                  entries,
                }).rid,
              };
            },

            createPipelineLayout(desc) {
              const layoutRids = (desc.bindGroupLayouts || []).map(
                (l) => l.__rid
              );
              return {
                __rid: gfx.op_gfx_device_create_pipeline_layout({
                  label: desc.label ?? null,
                  layout_rids: layoutRids,
                }).rid,
              };
            },

            createBindGroup(desc) {
              const entries = (desc.entries || []).map((e) => {
                const res = e.resource;
                let bufferRid = null,
                  offset = 0,
                  size = null,
                  samplerRid = null,
                  textureViewRid = null;
                if (res?.buffer) {
                  bufferRid = res.buffer.__rid ?? null;
                  offset = res.offset ?? 0;
                  size = res.size ?? null;
                } else if (res?.__isSampler) {
                  samplerRid = res.__rid;
                } else if (res?.__rid) {
                  textureViewRid = res.__rid;
                }
                return {
                  binding: e.binding >>> 0,
                  buffer_rid: bufferRid,
                  offset,
                  size,
                  sampler_rid: samplerRid,
                  texture_view_rid: textureViewRid,
                };
              });
              return {
                __rid: gfx.op_gfx_device_create_bind_group({
                  label: desc.label ?? null,
                  layout_rid: desc.layout.__rid,
                  entries,
                }).rid,
              };
            },

            createQuerySet(desc) {
              return { descriptor: desc, destroy() {} };
            },

            createCommandEncoder() {
              return new EmulatedEncoder(globalThis.__gpuCanvasContext);
            },

            createRenderBundleEncoder(descriptor) {
              const commands = [];
              let currentPipeline = null;
              const vertexBuffers = new Map();
              let indexBuffer = null,
                indexFormat = null;
              const bindGroups = new Map();

              return {
                setPipeline(p) {
                  currentPipeline = p;
                  commands.push({ type: "setPipeline", pipeline: p });
                },
                setVertexBuffer(slot, buf, offset = 0, size) {
                  vertexBuffers.set(slot, { buffer: buf, offset, size });
                },
                setIndexBuffer(buf, fmt, offset = 0, size) {
                  indexBuffer = buf;
                  indexFormat = fmt;
                },
                setBindGroup(idx, bg, offs) {
                  bindGroups.set(idx, bg);
                },
                draw(vc, ic = 1, fv = 0, fi = 0) {
                  commands.push({
                    type: "draw",
                    vertexCount: vc,
                    instanceCount: ic,
                    firstVertex: fv,
                    firstInstance: fi,
                    pipeline: currentPipeline,
                    vertexBuffers: new Map(vertexBuffers),
                    bindGroups: new Map(bindGroups),
                  });
                },
                drawIndexed(ic, inst = 1, fi = 0, bv = 0, fInst = 0) {
                  commands.push({
                    type: "drawIndexed",
                    indexCount: ic,
                    instanceCount: inst,
                    firstIndex: fi,
                    baseVertex: bv,
                    firstInstance: fInst,
                    pipeline: currentPipeline,
                    vertexBuffers: new Map(vertexBuffers),
                    bindGroups: new Map(bindGroups),
                    indexBuffer,
                    indexFormat,
                  });
                },
                finish() {
                  return { __isRenderBundle: true, commands };
                },
                setViewport() {},
                setScissorRect() {},
                setBlendConstant() {},
                setStencilReference() {},
                insertDebugMarker() {},
                pushDebugGroup() {},
                popDebugGroup() {},
              };
            },

            queue: {
              submit() {
                gfx.op_gfx_queue_submit_empty();
              },

              writeBuffer(buffer, dstOffset, data, dataOffset = 0, size) {
                if (buffer.__rid == null)
                  throw new Error("writeBuffer: no __rid");

                let srcView;
                if (data instanceof Uint8Array) {
                  srcView = data;
                } else if (
                  data instanceof ArrayBuffer ||
                  data instanceof SharedArrayBuffer
                ) {
                  srcView = new Uint8Array(data);
                } else if (ArrayBuffer.isView(data)) {
                  srcView = new Uint8Array(
                    data.buffer,
                    data.byteOffset,
                    data.byteLength
                  );
                } else {
                  throw new Error(`writeBuffer: unsupported data type`);
                }

                const byteOffset = dataOffset >>> 0;
                const byteLength =
                  size != null
                    ? size >>> 0
                    : (srcView.byteLength - byteOffset) >>> 0;

                gfx.op_gfx_queue_write_buffer(
                  buffer.__rid >>> 0,
                  dstOffset >>> 0,
                  new Uint8Array(
                    srcView.buffer,
                    srcView.byteOffset + byteOffset,
                    byteLength
                  )
                );
              },

              writeTexture(dest, data, layout, size) {
                const tex = dest.texture;
                if (!tex?.__rid) throw new Error("writeTexture: no __rid");
                const w = size.width ?? size[0] ?? 1;
                const h = size.height ?? size[1] ?? 1;
                if (w > tex.width || h > tex.height) return;
                const bytes =
                  data instanceof Uint8Array
                    ? data
                    : data instanceof ArrayBuffer
                    ? new Uint8Array(data)
                    : ArrayBuffer.isView(data)
                    ? new Uint8Array(
                        data.buffer,
                        data.byteOffset,
                        data.byteLength
                      )
                    : (() => {
                        throw new Error("writeTexture: unsupported");
                      })();
                gfx.op_gfx_write_texture_image(
                  tex.__rid,
                  w,
                  h,
                  dest.origin?.x ?? dest.origin?.[0] ?? 0,
                  dest.origin?.y ?? dest.origin?.[1] ?? 0,
                  dest.origin?.z ?? dest.origin?.[2] ?? 0,
                  dest.mipLevel ?? 0,
                  bytes
                );
              },

              copyExternalImageToTexture(source, dest, copySize) {
                const ib = source.source;
                if (!(ib instanceof ImageBitmap) || ib._rid == null)
                  throw new Error("ImageBitmap with rid required");
                if (!dest.texture?.__rid)
                  throw new Error("dest texture has no __rid");
                gfx.op_gfx_upload_decoded_image_to_texture(
                  ib._rid,
                  dest.texture.__rid,
                  dest.mipLevel ?? 0,
                  dest.origin?.x ?? dest.origin?.[0] ?? 0,
                  dest.origin?.y ?? dest.origin?.[1] ?? 0,
                  dest.origin?.z ?? dest.origin?.[2] ?? 0
                );
                gfx.op_gfx_decoded_image_drop(ib._rid);
                ib._rid = null;
              },
            },

            pushErrorScope() {},
            async popErrorScope() {
              return null;
            },
            destroy() {},
          };

          return device;
        },
      };
    },

    getPreferredCanvasFormat() {
      return (
        globalThis.__xrSessionInfo?.format ??
        gfx.op_gfx_get_preferred_surface_format()
      );
    },
  };

  Object.defineProperty(navigator, "gpu", {
    value: gpuShim,
    configurable: true,
    enumerable: true,
    writable: true,
  });

  console.log("[shims] WebGPU initialized");
}
