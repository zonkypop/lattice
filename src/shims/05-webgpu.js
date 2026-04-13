// 05-webgpu.js - WebGPU shims

const gfx = globalThis.__gfx;
if (!gfx) {
  console.log("[shims] WebGPU skipped (no native ops)");
} else {
  // Op timing profiler (disabled by default, enable via globalThis.__opTiming.enabled = true)
  const opTiming = {
    enabled: false,
    times: {},
    counts: {},
    frameCount: 0,
    LOG_INTERVAL: 200,
    track(name, fn) {
      if (!this.enabled) return fn();
      const t0 = performance.now();
      const result = fn();
      const elapsed = performance.now() - t0;
      this.times[name] = (this.times[name] || 0) + elapsed;
      this.counts[name] = (this.counts[name] || 0) + 1;
      return result;
    },
    endFrame() {
      if (!this.enabled) return;
      this.frameCount++;
      if (this.frameCount >= this.LOG_INTERVAL) {
        const entries = Object.entries(this.times)
          .map(([name, total]) => ({ name, total, count: this.counts[name], avg: total / this.frameCount }))
          .sort((a, b) => b.total - a.total)
          .slice(0, 5);
        const summary = entries.map(e => `${e.name}: ${e.avg.toFixed(2)}ms (${e.count}x)`).join(', ');
        console.log(`[OpTiming] ${summary}`);
        this.times = {};
        this.counts = {};
        this.frameCount = 0;
      }
    }
  };
  globalThis.__opTiming = opTiming;
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
  globalThis.GPUShaderStage = {
    VERTEX: 1,
    FRAGMENT: 2,
    COMPUTE: 4,
  };

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

    // Resize from an already-decoded ImageBitmap (pixel-level resize, no re-decode)
    if (source instanceof ImageBitmap) {
      if (source._closed || source._rid == null)
        throw new Error("ImageBitmap is closed");
      const needsResize = opts.resizeWidth && opts.resizeHeight;
      if (!needsResize) {
        // No resize requested — clone by resizing to same dimensions
        const { rid, width, height } = gfx.op_gfx_resize_decoded_image({
          source_rid: source._rid,
          width: source._width,
          height: source._height,
        });
        return new ImageBitmap(width, height, rid);
      }
      const { rid, width, height } = gfx.op_gfx_resize_decoded_image({
        source_rid: source._rid,
        width: opts.resizeWidth,
        height: opts.resizeHeight,
        quality: opts.resizeQuality ?? null,
      });
      return new ImageBitmap(width, height, rid);
    }

    if (!(source instanceof Blob))
      throw new Error("Only Blob and ImageBitmap sources supported");

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

    const { rid, width, height } = await gfx.op_gfx_decode_image_store(bytes, {
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

    __submitFrame(timestampWrites, resolveQueries, copyBuffers, computeCommands) {
      if (this._frameSubmitted || globalThis.__xrMode) {
        this._pendingDrawCalls = [];
        this._frameSubmitted = true;
        return;
      }
      if (this._pendingDrawCalls.length) {
        const args = { draw_calls: this._pendingDrawCalls };

        // Add compute commands to execute BEFORE render pass (all in one submission)
        if (computeCommands && computeCommands.length > 0) {
          args.compute_commands = computeCommands;
        }

        // Add timestamp writes if provided
        if (timestampWrites) {
          args.timestamp_writes = timestampWrites;
        }

        // Add resolve and copy operations if provided
        if (resolveQueries && resolveQueries.length > 0) {
          args.resolve_query_sets = resolveQueries;
        }
        if (copyBuffers && copyBuffers.length > 0) {
          args.copy_buffers = copyBuffers;
        }

        gfx.op_gfx_surface_draw(args);
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
    destroy() {
      if (this.__rid != null) {
        gfx.op_gfx_buffer_drop(this.__rid);
        this.__rid = null;
      }
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
        usage: viewDesc.usage ?? null,
        base_mip_level: viewDesc.baseMipLevel ?? null,
        mip_level_count: viewDesc.mipLevelCount ?? null,
        base_array_layer: viewDesc.baseArrayLayer ?? null,
        array_layer_count: viewDesc.arrayLayerCount ?? null,
      });
      return {
        __rid: v.rid,
        destroy() {
          if (this.__rid != null) {
            gfx.op_gfx_resource_drop(this.__rid);
            this.__rid = null;
          }
        },
      };
    }

    destroy() {
      if (this.__rid != null) {
        gfx.op_gfx_texture_drop(this.__rid);
        this.__rid = null;
      }
    }
  }

  // Render pass
  class EmulatedRenderPass {
    constructor(context, desc, encoder) {
      this._context = context;
      this._encoder = encoder || null;
      this._pipelineRid = null;
      this._vertexBuffers = new Map();
      this._indexBuffer = null;
      this._indexFormat = null;
      this._bindGroups = new Map();
      this._drawCalls = [];
      this._stencilReference = 0;
      this._scissorRect = null;

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
      if (ca?.loadOp === "clear") {
        if (ca.clearValue) {
          const cv = ca.clearValue;
          this._clearColor = Array.isArray(cv)
            ? { r: cv[0] ?? 0, g: cv[1] ?? 0, b: cv[2] ?? 0, a: cv[3] ?? 1 }
            : { r: cv.r ?? 0, g: cv.g ?? 0, b: cv.b ?? 0, a: cv.a ?? 1 };
        } else {
          this._clearColor = { r: 0, g: 0, b: 0, a: 0 };
        }
      }

      // Capture color attachment store op
      this._colorStoreOp = ca?.storeOp || "store";

      const da = desc?.depthStencilAttachment;
      this._depthAttachment =
        da?.view?.__rid != null
          ? {
              viewRid: da.view.__rid,
              depthClearValue: da.depthClearValue ?? 1.0,
              depthLoadOp: da.depthLoadOp || "clear",
              depthStoreOp: da.depthStoreOp || "store",
              depthReadOnly: !!da.depthReadOnly,
              stencilClearValue: da.stencilClearValue ?? 0,
              stencilLoadOp: da.stencilLoadOp || null,
              stencilStoreOp: da.stencilStoreOp || null,
              stencilReadOnly: !!da.stencilReadOnly,
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
        stencil_clear_value: isFirst
          ? this._depthAttachment?.stencilClearValue ?? null
          : null,
        stencil_load_op: isFirst
          ? this._depthAttachment?.stencilLoadOp ?? null
          : null,
        stencil_store_op: isFirst
          ? this._depthAttachment?.stencilStoreOp ?? null
          : null,
        stencil_read_only: isFirst
          ? this._depthAttachment?.stencilReadOnly ?? null
          : null,
        stencil_reference: this._stencilReference || null,
        scissor_rect: this._scissorRect,
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
    setScissorRect(x, y, w, h) { this._scissorRect = [x >>> 0, y >>> 0, w >>> 0, h >>> 0]; }
    setBlendConstant() {}
    setStencilReference(ref) { this._stencilReference = ref; }
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
        stencil_clear_value: isFirst
          ? this._depthAttachment?.stencilClearValue ?? null
          : null,
        stencil_load_op: isFirst
          ? this._depthAttachment?.stencilLoadOp ?? null
          : null,
        stencil_store_op: isFirst
          ? this._depthAttachment?.stencilStoreOp ?? null
          : null,
        stencil_read_only: isFirst
          ? this._depthAttachment?.stencilReadOnly ?? null
          : null,
        stencil_reference: this._stencilReference || null,
        scissor_rect: this._scissorRect,
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
        stencil_clear_value: isFirst
          ? this._depthAttachment?.stencilClearValue ?? null
          : null,
        stencil_load_op: isFirst
          ? this._depthAttachment?.stencilLoadOp ?? null
          : null,
        stencil_store_op: isFirst
          ? this._depthAttachment?.stencilStoreOp ?? null
          : null,
        stencil_read_only: isFirst
          ? this._depthAttachment?.stencilReadOnly ?? null
          : null,
        stencil_reference: this._stencilReference || null,
        scissor_rect: this._scissorRect,
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

    _flushEncoderCompute() {
      // Flush any pending compute commands from the parent encoder before
      // submitting a render op. This ensures compute dispatches that precede
      // this render pass (e.g. shadow index compaction) execute first.
      const enc = this._encoder;
      if (enc && enc._commands.length > 0) {
        gfx.op_gfx_compute_batch({ commands: enc._commands });
        enc._commands.length = 0;
      }
    }

    end() {
      if (!this._drawCalls.length) return;

      if (this._isDepthOnlyPass && this._depthAttachment?.viewRid != null) {
        this._flushEncoderCompute();
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
          this._flushEncoderCompute();
          gfx.op_gfx_render_xr_frame({
            msaa_view_rid: msaaViewRid,
            resolve_view_rid: resolve.__rid,
            draw_calls: this._drawCalls,
            depth_view_rid: this._depthAttachment?.viewRid ?? null,
            depth_clear_value: this._depthAttachment?.depthClearValue ?? 1.0,
            depth_store_op: this._depthAttachment?.depthStoreOp ?? "store",
            color_store_op: this._colorStoreOp,
            clear_color: this._clearColor,
          });
        }
        return;
      }

      if (this._isRenderToTexture && this._colorTargetView?.__rid != null) {
        this._flushEncoderCompute();
        gfx.op_gfx_render_to_texture({
          target_view_rid: this._colorTargetView.__rid,
          draw_calls: this._drawCalls,
        });
        return;
      }

      this._context?.__queueDrawCalls?.(this._drawCalls);
    }
  }

  // Compute pass
  class EmulatedComputePass {
    constructor(encoder, desc) {
      this._encoder = encoder;
      this._pipelineRid = null;
      this._bindGroups = new Map();
      // Capture timestampWrites from descriptor
      this._timestampWrites = null;
      if (desc?.timestampWrites) {
        const tw = desc.timestampWrites;
        if (tw.querySet?.__rid != null) {
          this._timestampWrites = {
            query_set_rid: tw.querySet.__rid,
            beginning_of_pass_write_index: tw.beginningOfPassWriteIndex ?? null,
            end_of_pass_write_index: tw.endOfPassWriteIndex ?? null,
          };
        }
      }
    }

    setPipeline(p) {
      this._pipelineRid = p.__rid;
      this._isComputePipeline = p.__isComputePipeline === true;
    }

    setBindGroup(index, bg) {
      if (bg?.__rid != null) this._bindGroups.set(index >>> 0, bg.__rid);
    }

    dispatchWorkgroups(x, y = 1, z = 1) {
      if (this._pipelineRid == null) return;

      const bgRids = [];
      for (const [idx, rid] of this._bindGroups) bgRids[idx] = rid;

      const cmd = {
        cmd: "dispatch",
        pipeline_rid: this._pipelineRid,
        bind_group_rids: bgRids,
        workgroup_count_x: x >>> 0,
        workgroup_count_y: y >>> 0,
        workgroup_count_z: z >>> 0,
      };
      if (this._timestampWrites) {
        cmd.timestamp_writes = this._timestampWrites;
      }
      this._encoder._commands.push(cmd);
    }

    dispatchWorkgroupsIndirect(indirectBuffer, indirectOffset) {
      if (this._pipelineRid == null) return;
      if (!indirectBuffer?.__rid)
        throw new Error("dispatchWorkgroupsIndirect: no indirect buffer");

      const bgRids = [];
      for (const [idx, rid] of this._bindGroups) bgRids[idx] = rid;

      const cmd = {
        cmd: "dispatch_indirect",
        pipeline_rid: this._pipelineRid,
        bind_group_rids: bgRids,
        indirect_buffer_rid: indirectBuffer.__rid,
        indirect_offset: indirectOffset >>> 0,
      };
      if (this._timestampWrites) {
        cmd.timestamp_writes = this._timestampWrites;
      }
      this._encoder._commands.push(cmd);
    }

    insertDebugMarker() {}
    pushDebugGroup() {}
    popDebugGroup() {}

    end() {
      // Commands already pushed to encoder._commands during dispatch calls
    }
  }

  // Command encoder
  class EmulatedEncoder {
    constructor(context) {
      this._context = context;
      this._commands = [];
      this._timestampWrites = [];
      this._resolveQueries = [];
      this._copyBuffers = [];
    }

    beginRenderPass(desc) {
      return new EmulatedRenderPass(this._context, desc, this);
    }

    beginComputePass(desc) {
      return new EmulatedComputePass(this, desc);
    }

    writeTimestamp(querySet, queryIndex) {
      if (!querySet?.__rid) return;

      if (globalThis.__xrMode) {
        // XR mode: write timestamp immediately since XR render submits separately.
        // This ensures timestamps bracket the XR frame correctly.
        gfx.op_gfx_write_timestamp(querySet.__rid, queryIndex >>> 0);
      } else {
        // Non-XR: track timestamp writes for batching with render pass.
        this._timestampWrites.push({
          query_set_rid: querySet.__rid,
          query_index: queryIndex >>> 0,
        });
      }
    }

    resolveQuerySet(querySet, firstQuery, queryCount, destination, destinationOffset) {
      if (!querySet?.__rid || !destination?.__rid) return;
      this._resolveQueries.push({
        query_set_rid: querySet.__rid,
        first_query: firstQuery >>> 0,
        query_count: queryCount >>> 0,
        destination_rid: destination.__rid,
        destination_offset: destinationOffset >>> 0,
      });
    }

    copyBufferToBuffer(source, sourceOffset, destination, destinationOffset, size) {
      if (!source?.__rid || !destination?.__rid) return;
      this._copyBuffers.push({
        src_rid: source.__rid,
        src_offset: sourceOffset >>> 0,
        dst_rid: destination.__rid,
        dst_offset: destinationOffset >>> 0,
        size: size >>> 0,
      });
    }

    copyBufferToTexture(source, dest, copySize) {
      if (!source.buffer?.__rid) {
        throw new Error("copyBufferToTexture: buffer missing __rid");
      }
      if (!dest.texture?.__rid) {
        throw new Error("copyBufferToTexture: texture missing __rid");
      }
      const d = dest.origin || {};
      const width = copySize?.width ?? copySize?.[0] ?? 1;
      const height = copySize?.height ?? copySize?.[1] ?? 1;
      const depth = copySize?.depthOrArrayLayers ?? copySize?.[2] ?? 1;
      let bytesPerRow = source.bytesPerRow;
      if (!bytesPerRow || bytesPerRow < width) {
        bytesPerRow = Math.ceil(width / 256) * 256;
        if (bytesPerRow < 256) bytesPerRow = 256;
      }
      this._commands.push({
        cmd: "copy_buffer_to_texture",
        src_buffer_rid: source.buffer.__rid,
        buffer_offset: source.offset ?? 0,
        bytes_per_row: bytesPerRow,
        rows_per_image: source.rowsPerImage ?? height,
        texture_rid: dest.texture.__rid,
        mip_level: dest.mipLevel ?? 0,
        origin_x: d.x ?? 0,
        origin_y: d.y ?? 0,
        origin_z: d.z ?? 0,
        width,
        height,
        depth_or_array_layers: depth,
      });
    }

    copyTextureToTexture(src, dst, copySize) {
      if (!src.texture?.__rid || !dst.texture?.__rid) {
        throw new Error("copyTextureToTexture: texture missing __rid");
      }
      const so = src.origin || {};
      const d = dst.origin || {};
      this._commands.push({
        cmd: "copy_texture_to_texture",
        src_texture_rid: src.texture.__rid,
        src_mip_level: src.mipLevel ?? 0,
        src_origin_x: so.x ?? 0,
        src_origin_y: so.y ?? 0,
        src_origin_z: so.z ?? 0,
        texture_rid: dst.texture.__rid,
        mip_level: dst.mipLevel ?? 0,
        origin_x: d.x ?? 0,
        origin_y: d.y ?? 0,
        origin_z: d.z ?? 0,
        width: copySize?.width ?? copySize?.[0] ?? 1,
        height: copySize?.height ?? copySize?.[1] ?? 1,
        depth_or_array_layers:
          copySize?.depthOrArrayLayers ?? copySize?.[2] ?? 1,
      });
    }

    copyTextureToBuffer() {}

    clearBuffer(buffer, offset = 0, size) {
      if (!buffer?.__rid) throw new Error("clearBuffer: buffer missing __rid");
      const clearSize = size ?? buffer.byteLength - offset;
      this._commands.push({
        cmd: "clear_buffer",
        buffer_rid: buffer.__rid,
        offset: offset >>> 0,
        size: clearSize >>> 0,
      });
    }

    finish() {
      // Check if this encoder has pending timestamp resolve/copy operations
      const hasTimestampOps =
        this._resolveQueries.length > 0 || this._copyBuffers.length > 0;

      // Check if there's a canvas context with pending draw calls (render frame)
      const ctx = globalThis.__gpuCanvasContext;
      const hasRenderFrame = ctx && ctx._pendingDrawCalls?.length > 0;
      const isXrMode = globalThis.__xrMode;

      if (isXrMode) {
        // XR mode: render pass is handled separately via op_gfx_render_xr_frame.
        // Timestamps are written immediately in writeTimestamp() to bracket XR render.
        // We still need to submit compute commands and process resolve/copy ops.
        if (this._commands.length > 0) {
          gfx.op_gfx_compute_batch({ commands: this._commands });
        }
        // Process resolve and copy operations (timestamps already written immediately)
        if (hasTimestampOps) {
          gfx.op_gfx_timestamp_batch({
            write_timestamps: [],
            resolve_query_sets: this._resolveQueries,
            copy_buffers: this._copyBuffers,
          });
        }
        opTiming.endFrame();
      } else if (hasRenderFrame) {
        // Build render pass timestamp writes from encoder-level writeTimestamp calls.
        // These get passed to the native render pass for accurate GPU execution timing.
        let renderTimestampWrites = null;
        if (this._timestampWrites.length >= 2) {
          const startTs = this._timestampWrites.find((t) => t.query_index === 0);
          const endTs = this._timestampWrites.find((t) => t.query_index === 1);
          if (startTs && endTs && startTs.query_set_rid === endTs.query_set_rid) {
            renderTimestampWrites = {
              query_set_rid: startTs.query_set_rid,
              beginning_of_pass_write_index: 0,
              end_of_pass_write_index: 1,
            };
          }
        }

        // Pass compute commands to be executed in the same submission as render
        // This ensures all GPU work is measured by the frame timestamps
        ctx.__submitFrame(
          renderTimestampWrites,
          this._resolveQueries,
          this._copyBuffers,
          this._commands
        );
      } else if (this._commands.length > 0) {
        // Compute-only encoder (no render frame) - submit compute separately
        gfx.op_gfx_compute_batch({ commands: this._commands });

        if (hasTimestampOps) {
          gfx.op_gfx_timestamp_batch({
            write_timestamps: [],
            resolve_query_sets: this._resolveQueries,
            copy_buffers: this._copyBuffers,
          });
        }
      } else if (hasTimestampOps) {
        // No compute, no render, but has timestamp ops
        gfx.op_gfx_timestamp_batch({
          write_timestamps: [],
          resolve_query_sets: this._resolveQueries,
          copy_buffers: this._copyBuffers,
        });
      }

      // Clean up deferred GPU resources (e.g. compute mipmap views/bind groups
      // that must survive until commands are submitted above)
      if (this._deferredCleanup) {
        for (const r of this._deferredCleanup) if (r.destroy) r.destroy();
        this._deferredCleanup = null;
      }

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

  // QuerySet class for timestamp queries
  class EmulatedQuerySet {
    constructor(rid, type, count) {
      this.__rid = rid;
      this.type = type;
      this.count = count;
    }

    destroy() {
      if (this.__rid != null) {
        try {
          gfx.op_gfx_query_set_destroy(this.__rid);
        } catch (e) {
          // Ignore errors on destroy
        }
        this.__rid = null;
      }
    }
  }

  // Mappable buffer class for buffer.mapAsync support
  class MappableBuffer {
    constructor(rid, size, usage) {
      this.__rid = rid;
      this.byteLength = size;
      this.usage = usage;
      this._mapState = "unmapped";
      this._mappedOffset = 0;
      this._mappedSize = 0;
    }

    async mapAsync(mode, offset = 0, size) {
      if (this._mapState !== "unmapped") {
        throw new Error("Buffer already mapped or mapping pending");
      }
      const mapSize = size ?? (this.byteLength - offset);
      this._mapState = "pending";
      try {
        // Start the async map operation (non-blocking)
        gfx.op_gfx_buffer_map_async(this.__rid, mode, offset, mapSize);

        // Poll until ready with cheap yields (non-blocking)
        while (!gfx.op_gfx_buffer_map_poll(this.__rid)) {
          await globalThis.__yieldToRuntime();
        }

        // Finish the mapping
        gfx.op_gfx_buffer_map_finish(this.__rid);

        this._mapState = "mapped";
        this._mappedOffset = offset;
        this._mappedSize = mapSize;
      } catch (e) {
        this._mapState = "unmapped";
        throw e;
      }
    }

    getMappedRange(offset = 0, size) {
      if (this._mapState !== "mapped") {
        throw new Error("Buffer is not mapped");
      }
      const actualOffset = this._mappedOffset + offset;
      const actualSize = size ?? (this._mappedSize - offset);
      // Get the data from native
      const data = gfx.op_gfx_buffer_get_mapped_range(
        this.__rid,
        actualOffset,
        actualSize
      );
      // Return as ArrayBuffer
      return data.buffer.slice(
        data.byteOffset,
        data.byteOffset + data.byteLength
      );
    }

    unmap() {
      if (this._mapState === "mapped") {
        gfx.op_gfx_buffer_unmap(this.__rid);
      }
      this._mapState = "unmapped";
      this._mappedOffset = 0;
      this._mappedSize = 0;
    }

    // Low-level sync helpers for per-frame polling (avoids async spinning).
    // Usage: _startMap → (next frame) _pollMap → if true: _finishMap → getMappedRange → unmap
    _startMap(mode, offset, size) {
      gfx.op_gfx_buffer_map_async(this.__rid, mode, offset, size);
    }
    _pollMap() {
      return gfx.op_gfx_buffer_map_poll(this.__rid);
    }
    _finishMap(offset, size) {
      gfx.op_gfx_buffer_map_finish(this.__rid);
      this._mapState = "mapped";
      this._mappedOffset = offset;
      this._mappedSize = size;
    }

    destroy() {
      if (this.__rid != null) {
        gfx.op_gfx_buffer_drop(this.__rid);
        this.__rid = null;
      }
    }
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

      // Check which features are available
      const availableFeatures = [];
      let timestampPeriod = 1.0; // nanoseconds per tick (1.0 = ticks are already ns)
      try {
        const hasTimestamp = gfx.op_gfx_has_timestamp_query();
        console.log("[shims] Native timestamp-query support:", hasTimestamp);
        if (hasTimestamp) {
          availableFeatures.push("timestamp-query");
          timestampPeriod = gfx.op_gfx_get_timestamp_period();
          console.log("[shims] Timestamp period (ns/tick):", timestampPeriod);
        }
      } catch (e) {
        console.log("[shims] Feature check error:", e.message);
        // Feature check not available yet (before gfx context init)
      }
      // Expose timestamp period globally for GPUTimer to use
      globalThis.__gpuTimestampPeriod = timestampPeriod;

      const features = makeFeaturesSet(availableFeatures);
      const limits = {
        maxTextureDimension1D: 8192,
        maxTextureDimension2D: 8192,
        maxTextureDimension3D: 2048,
        maxTextureArrayLayers: 2048,
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
                depth_bias: depth?.depthBias ?? null,
                depth_bias_slope_scale: depth?.depthBiasSlopeScale ?? null,
                depth_bias_clamp: depth?.depthBiasClamp ?? null,
                stencil_front_compare: depth?.stencilFront?.compare ?? null,
                stencil_front_pass_op: depth?.stencilFront?.passOp ?? null,
                stencil_front_fail_op: depth?.stencilFront?.failOp ?? null,
                stencil_front_depth_fail_op: depth?.stencilFront?.depthFailOp ?? null,
                stencil_back_compare: depth?.stencilBack?.compare ?? null,
                stencil_back_pass_op: depth?.stencilBack?.passOp ?? null,
                stencil_back_fail_op: depth?.stencilBack?.failOp ?? null,
                stencil_back_depth_fail_op: depth?.stencilBack?.depthFailOp ?? null,
                stencil_read_mask: depth?.stencilReadMask ?? null,
                stencil_write_mask: depth?.stencilWriteMask ?? null,
                color_format: desc.fragment?.targets?.[0]?.format ?? null,
                color_write_mask: desc.fragment?.targets?.[0]?.writeMask ?? null,
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

            createComputePipeline(desc) {
              const rid = gfx.op_gfx_device_create_compute_pipeline({
                shader_module_rid: desc.compute.module.__rid,
                entry_point: desc.compute.entryPoint ?? "main",
                pipeline_layout_rid:
                  desc.layout === "auto" ? null : desc.layout?.__rid ?? null,
              }).rid;

              return {
                __rid: rid,
                __isComputePipeline: true,
                getBindGroupLayout(index) {
                  return {
                    __rid: gfx.op_gfx_compute_pipeline_get_bind_group_layout({
                      pipeline_rid: rid,
                      index: index >>> 0,
                    }).rid,
                  };
                },
              };
            },

            async createComputePipelineAsync(desc) {
              // Synchronous implementation wrapped in Promise for API compatibility
              return this.createComputePipeline(desc);
            },

            createBuffer(options) {
              const size = options.size >>> 0;
              const usage = options.usage >>> 0;
              if (options.mappedAtCreation)
                return new EmulatedBuffer(size, usage);
              const rid = gfx.op_gfx_device_create_buffer({
                size,
                usage,
                label: options.label ?? null,
              }).rid;
              // Use MappableBuffer for buffers with MAP_READ or MAP_WRITE usage
              if (usage & (GPUBufferUsage.MAP_READ | GPUBufferUsage.MAP_WRITE)) {
                return new MappableBuffer(rid, size, usage);
              }
              return {
                __rid: rid,
                byteLength: size,
                usage,
                destroy() {
                  if (this.__rid != null) {
                    gfx.op_gfx_buffer_drop(this.__rid);
                    this.__rid = null;
                  }
                },
              };
            },

            // Fast path: create a GPU buffer and upload data in a single op,
            // avoiding the EmulatedBuffer double-copy (JS alloc + set + unmap).
            createBufferWithData(data, usage) {
              const u = usage >>> 0;
              const bytes = data instanceof Uint8Array
                ? data
                : new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
              const rid = gfx.op_gfx_device_create_buffer_init(u, null, bytes).rid;
              return {
                __rid: rid,
                byteLength: bytes.byteLength,
                usage: u,
                destroy() {
                  if (this.__rid != null) {
                    gfx.op_gfx_buffer_drop(this.__rid);
                    this.__rid = null;
                  }
                },
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
                usage: (desc.usage >>> 0),
                view_formats: desc.viewFormats || [],
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
              const rid = gfx.op_gfx_device_create_bind_group({
                  label: desc.label ?? null,
                  layout_rid: desc.layout.__rid,
                  entries,
                }).rid;
              return {
                __rid: rid,
                destroy() {
                  if (this.__rid != null) {
                    gfx.op_gfx_bind_group_drop(this.__rid);
                    this.__rid = null;
                  }
                },
              };
            },

            createQuerySet(desc) {
              // For timestamp queries, use native implementation
              if (desc.type === "timestamp") {
                try {
                  const result = gfx.op_gfx_device_create_query_set({
                    query_type: desc.type,
                    count: desc.count >>> 0,
                  });
                  return new EmulatedQuerySet(result.rid, desc.type, desc.count);
                } catch (e) {
                  console.warn("Failed to create timestamp query set:", e);
                  // Fall back to stub
                  return { descriptor: desc, destroy() {} };
                }
              }
              // Other query types not yet supported
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

                const rid = buffer.__rid >>> 0;
                const dst = dstOffset >>> 0;

                // Fast path: Uint8Array with no sub-range, pass directly
                if (data instanceof Uint8Array && dataOffset === 0 && size == null) {
                  gfx.op_gfx_queue_write_buffer(rid, dst, data);
                  return;
                }

                let bytes;
                if (data instanceof ArrayBuffer || data instanceof SharedArrayBuffer) {
                  const off = dataOffset >>> 0;
                  const len = size != null
                    ? size >>> 0
                    : data.byteLength - off;
                  bytes = new Uint8Array(data, off, len);
                } else if (ArrayBuffer.isView(data)) {
                  const elemSize = data.BYTES_PER_ELEMENT || 1;
                  const off = (dataOffset >>> 0) * elemSize;
                  const len = size != null
                    ? (size >>> 0) * elemSize
                    : data.byteLength - off;
                  bytes = new Uint8Array(data.buffer, data.byteOffset + off, len);
                } else {
                  throw new Error(`writeBuffer: unsupported data type`);
                }

                gfx.op_gfx_queue_write_buffer(rid, dst, bytes);
              },

              writeTexture(dest, data, layout, size) {
                const tex = dest.texture;
                if (!tex?.__rid) throw new Error("writeTexture: no __rid");
                const w = size.width ?? size[0] ?? 1;
                const h = size.height ?? size[1] ?? 1;
                const d = size.depthOrArrayLayers ?? size[2] ?? 1;
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
                  d,
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
