// 06-webxr.js - WebXR shims for native OpenXR

// Check for native XR ops - access directly, not cached
const isNativeXR =
  globalThis.__xr && typeof globalThis.__xr.op_xr_is_supported === "function";

if (!isNativeXR) {
  console.log("[shims] WebXR skipped (no native ops)");
} else {
  const { SimpleEventTarget } = globalThis;

  // DOMPointReadOnly polyfill
  if (typeof DOMPointReadOnly === "undefined") {
    globalThis.DOMPointReadOnly = class DOMPointReadOnly {
      constructor(x = 0, y = 0, z = 0, w = 1) {
        this.x = x;
        this.y = y;
        this.z = z;
        this.w = w;
      }
      static fromPoint(o) {
        return new DOMPointReadOnly(o.x, o.y, o.z, o.w);
      }
      toJSON() {
        return { x: this.x, y: this.y, z: this.z, w: this.w };
      }
    };
  }

  let __xrSession = null;

  // Storage for grip connected callbacks (Three.js WebXRManager compatibility)
  globalThis.__xrGripConnectedCallbacks = [[], []];
  globalThis.__xrControllerConnectedCallbacks = [[], []];

  // XRRigidTransform
  class XRRigidTransform {
    constructor(position, orientation) {
      const p = position || {};
      const o = orientation || {};
      this.position = new DOMPointReadOnly(
        Array.isArray(p) ? p[0] : p.x || 0,
        Array.isArray(p) ? p[1] : p.y || 0,
        Array.isArray(p) ? p[2] : p.z || 0,
        1
      );
      this.orientation = new DOMPointReadOnly(
        Array.isArray(o) ? o[0] : o.x || 0,
        Array.isArray(o) ? o[1] : o.y || 0,
        Array.isArray(o) ? o[2] : o.z || 0,
        Array.isArray(o) ? o[3] : o.w || 1
      );
      this._matrix = null;
      this._inverse = null;
    }

    get matrix() {
      if (!this._matrix) {
        const mat = new Float32Array(16);
        const { x, y, z, w } = this.orientation;
        const x2 = x + x,
          y2 = y + y,
          z2 = z + z;
        const xx = x * x2,
          xy = x * y2,
          xz = x * z2;
        const yy = y * y2,
          yz = y * z2,
          zz = z * z2;
        const wx = w * x2,
          wy = w * y2,
          wz = w * z2;
        mat[0] = 1 - (yy + zz);
        mat[1] = xy + wz;
        mat[2] = xz - wy;
        mat[3] = 0;
        mat[4] = xy - wz;
        mat[5] = 1 - (xx + zz);
        mat[6] = yz + wx;
        mat[7] = 0;
        mat[8] = xz + wy;
        mat[9] = yz - wx;
        mat[10] = 1 - (xx + yy);
        mat[11] = 0;
        mat[12] = this.position.x;
        mat[13] = this.position.y;
        mat[14] = this.position.z;
        mat[15] = 1;
        this._matrix = mat;
      }
      return this._matrix;
    }

    get inverse() {
      if (!this._inverse) {
        const { x, y, z, w } = this.orientation;
        const invQ = { x: -x, y: -y, z: -z, w };
        const vx = -this.position.x,
          vy = -this.position.y,
          vz = -this.position.z;
        const tx = 2 * (invQ.y * vz - invQ.z * vy);
        const ty = 2 * (invQ.z * vx - invQ.x * vz);
        const tz = 2 * (invQ.x * vy - invQ.y * vx);
        this._inverse = new XRRigidTransform(
          {
            x: vx + invQ.w * tx + invQ.y * tz - invQ.z * ty,
            y: vy + invQ.w * ty + invQ.z * tx - invQ.x * tz,
            z: vz + invQ.w * tz + invQ.x * ty - invQ.y * tx,
          },
          invQ
        );
      }
      return this._inverse;
    }
  }

  // XRSpace - base class for all spaces
  class XRSpace extends SimpleEventTarget {
    constructor() {
      super();
    }
  }

  // XRReferenceSpace
  class XRReferenceSpace extends XRSpace {
    constructor(type) {
      super();
      this._type = type;
      this._transform = new XRRigidTransform();
    }
    getOffsetReferenceSpace(offset) {
      const space = new XRReferenceSpace(this._type);
      space._transform = offset;
      return space;
    }
  }

  class XRBoundedReferenceSpace extends XRReferenceSpace {
    constructor() {
      super("bounded-floor");
      this._boundsGeometry = [];
    }
    get boundsGeometry() {
      return this._boundsGeometry;
    }
  }

  // XRInputSource
  class XRInputSource {
    constructor(data) {
      this._handedness = data.handedness || "none";
      this._targetRayMode = data.target_ray_mode || "tracked-pointer";
      this._profiles = data.profiles || [];
      this._gripSpace = data.grip_space_pose
        ? new XRInputSourceSpace("grip", data)
        : null;
      this._targetRaySpace = data.target_ray_pose
        ? new XRInputSourceSpace("target-ray", data)
        : null;
      this._hand = data.hand ? new XRHand(data.hand) : null;
      this._gamepad = data.gamepad ? new XRGamepad(data.gamepad) : null;

      // Internal tracking
      this._connected = false;
      this._selectPressed = false;
      this._squeezePressed = false;
      this._pinching = false;
    }

    get handedness() {
      return this._handedness;
    }
    get targetRayMode() {
      return this._targetRayMode;
    }
    get targetRaySpace() {
      return this._targetRaySpace;
    }
    get gripSpace() {
      return this._gripSpace;
    }
    get profiles() {
      return this._profiles;
    }
    get gamepad() {
      return this._gamepad;
    }
    get hand() {
      return this._hand;
    }

    _update(data) {
      if (this._gripSpace && data.grip_space_pose) {
        this._gripSpace._updatePose(data.grip_space_pose);
      }
      if (this._targetRaySpace && data.target_ray_pose) {
        this._targetRaySpace._updatePose(data.target_ray_pose);
      }
      if (this._gamepad && data.gamepad) {
        this._gamepad._update(data.gamepad);
      }
      if (this._hand && data.hand) {
        this._hand._update(data.hand);
      }
    }
  }

  // XRInputSourceSpace - space for grip/target ray
  class XRInputSourceSpace extends XRSpace {
    constructor(type, data) {
      super();
      this._type = type;
      this._pose = null;
      const poseData =
        type === "grip" ? data.grip_space_pose : data.target_ray_pose;
      if (poseData) {
        this._updatePose(poseData);
      }
    }

    _updatePose(poseData) {
      this._pose = new XRRigidTransform(
        {
          x: poseData.position[0],
          y: poseData.position[1],
          z: poseData.position[2],
        },
        {
          x: poseData.orientation[0],
          y: poseData.orientation[1],
          z: poseData.orientation[2],
          w: poseData.orientation[3],
        }
      );
      if (poseData.matrix) {
        this._pose._matrix = new Float32Array(poseData.matrix);
      }
    }
  }

  // XRGamepad - mirrors standard Gamepad API
  class XRGamepad {
    constructor(data) {
      this.id = "";
      this.index = -1;
      this.connected = true;
      this.timestamp = performance.now();
      this.mapping = "xr-standard";
      this.axes = [0, 0, 0, 0]; // Use regular array, not Float32Array
      this.buttons = (data.buttons || []).map((b) => new XRGamepadButton(b));
      this.hapticActuators = [];
      this._update(data);
    }

    _update(data) {
      this.timestamp = performance.now();
      if (data.axes) {
        this.axes[0] = data.axes[0] || 0;
        this.axes[1] = data.axes[1] || 0;
        this.axes[2] = data.axes[2] || 0;
        this.axes[3] = data.axes[3] || 0;
      }
      if (data.buttons) {
        for (
          let i = 0;
          i < data.buttons.length && i < this.buttons.length;
          i++
        ) {
          this.buttons[i]._update(data.buttons[i]);
        }
      }
    }
  }

  // XRGamepadButton
  class XRGamepadButton {
    constructor(data) {
      this.pressed = data.pressed || false;
      this.touched = data.touched || false;
      this.value = data.value || 0;
    }

    _update(data) {
      this.pressed = data.pressed || false;
      this.touched = data.touched || false;
      this.value = data.value || 0;
    }
  }

  // XRHand - hand tracking
  class XRHand {
    constructor(data) {
      this._joints = new Map();
      this._jointNames = [
        "wrist",
        "thumb-metacarpal",
        "thumb-phalanx-proximal",
        "thumb-phalanx-distal",
        "thumb-tip",
        "index-finger-metacarpal",
        "index-finger-phalanx-proximal",
        "index-finger-phalanx-intermediate",
        "index-finger-phalanx-distal",
        "index-finger-tip",
        "middle-finger-metacarpal",
        "middle-finger-phalanx-proximal",
        "middle-finger-phalanx-intermediate",
        "middle-finger-phalanx-distal",
        "middle-finger-tip",
        "ring-finger-metacarpal",
        "ring-finger-phalanx-proximal",
        "ring-finger-phalanx-intermediate",
        "ring-finger-phalanx-distal",
        "ring-finger-tip",
        "pinky-finger-metacarpal",
        "pinky-finger-phalanx-proximal",
        "pinky-finger-phalanx-intermediate",
        "pinky-finger-phalanx-distal",
        "pinky-finger-tip",
      ];

      if (data && data.joints) {
        this._update(data);
      }
    }

    get size() {
      return this._joints.size;
    }

    get(jointName) {
      return this._joints.get(jointName) || null;
    }

    [Symbol.iterator]() {
      return this._joints[Symbol.iterator]();
    }

    entries() {
      return this._joints.entries();
    }
    keys() {
      return this._joints.keys();
    }
    values() {
      return this._joints.values();
    }
    forEach(fn) {
      this._joints.forEach(fn);
    }

    _update(data) {
      if (!data.joints) return;

      for (const joint of data.joints) {
        let jointSpace = this._joints.get(joint.name);
        if (!jointSpace) {
          jointSpace = new XRJointSpace(joint);
          this._joints.set(joint.name, jointSpace);
        } else {
          jointSpace._update(joint);
        }
      }
    }
  }

  // XRPose / XRViewerPose - MUST be defined before XRJointPose
  class XRPose {
    constructor(transform, emulated = false) {
      this.transform = transform;
      this.emulatedPosition = emulated;
      this.linearVelocity = null;
      this.angularVelocity = null;
    }
  }

  // XRJointSpace
  class XRJointSpace extends XRSpace {
    constructor(data) {
      super();
      this.jointName = data.name;
      this._radius = data.radius || 0;
      this._pose = null;
      this._update(data);
    }

    get radius() {
      return this._radius;
    }

    _update(data) {
      this._radius = data.radius || this._radius;
      if (data.pose) {
        this._pose = new XRRigidTransform(
          {
            x: data.pose.position[0],
            y: data.pose.position[1],
            z: data.pose.position[2],
          },
          {
            x: data.pose.orientation[0],
            y: data.pose.orientation[1],
            z: data.pose.orientation[2],
            w: data.pose.orientation[3],
          }
        );
        if (data.pose.matrix) {
          this._pose._matrix = new Float32Array(data.pose.matrix);
        }
      }
    }
  }

  // XRJointPose
  class XRJointPose extends XRPose {
    constructor(transform, radius) {
      super(transform);
      this.radius = radius;
    }
  }

  // XRInputSourceArray
  class XRInputSourceArray {
    constructor() {
      this._sources = [];
    }
    get length() {
      return this._sources.length;
    }
    [Symbol.iterator]() {
      return this._sources[Symbol.iterator]();
    }
    entries() {
      return this._sources.entries();
    }
    keys() {
      return this._sources.keys();
    }
    values() {
      return this._sources.values();
    }
    forEach(fn) {
      this._sources.forEach(fn);
    }

    _updateFromNative(sourcesData) {
      // Track which sources we've seen
      const seenHandedness = new Set();

      for (const data of sourcesData) {
        seenHandedness.add(data.handedness);

        // Find existing source
        let source = this._sources.find(
          (s) => s.handedness === data.handedness
        );

        if (source) {
          // Update existing
          source._update(data);
        } else {
          // Create new
          source = new XRInputSource(data);
          this._sources.push(source);
        }
      }

      // Remove sources that are no longer present
      this._sources = this._sources.filter((s) =>
        seenHandedness.has(s.handedness)
      );
    }
  }

  // XRView
  class XRView {
    constructor(data) {
      this.eye = data.view_index === 0 ? "left" : "right";
      this.projectionMatrix = new Float32Array(data.projection_matrix);
      this.transform = new XRRigidTransform(
        {
          x: data.transform.position[0],
          y: data.transform.position[1],
          z: data.transform.position[2],
        },
        {
          x: data.transform.orientation[0],
          y: data.transform.orientation[1],
          z: data.transform.orientation[2],
          w: data.transform.orientation[3],
        }
      );
      this.transform._matrix = new Float32Array(data.transform.matrix);
      this._viewIndex = data.view_index;
      this._viewport = {
        x: data.viewport_x || 0,
        y: data.viewport_y || 0,
        width: data.viewport_width,
        height: data.viewport_height,
      };
      this.recommendedViewportScale = 1.0;
    }
    requestViewportScale(scale) {
      this.recommendedViewportScale = Math.max(0.25, Math.min(1.0, scale));
    }
  }

  class XRViewerPose {
    constructor(data) {
      if (data.views?.length) {
        const v = data.views[0];
        this.transform = new XRRigidTransform(
          {
            x: v.transform.position[0],
            y: v.transform.position[1],
            z: v.transform.position[2],
          },
          {
            x: v.transform.orientation[0],
            y: v.transform.orientation[1],
            z: v.transform.orientation[2],
            w: v.transform.orientation[3],
          }
        );
      } else {
        this.transform = new XRRigidTransform();
      }
      this.views = data.views.map((v) => new XRView(v));
      this.emulatedPosition = false;
    }
  }

  // XRFrame
  class XRFrame {
    constructor(session, state) {
      this.session = session;
      this._state = state;
      this._pose = null;
      this._cachedPose = null;
      this._viewSubImageCache = new Map();
    }
    get predictedDisplayTime() {
      return this._state.predicted_display_time;
    }
    getViewerPose() {
      if (!this._pose) {
        const data =
          this._cachedPose || globalThis.__xr.op_xr_get_viewer_pose();
        this._pose = new XRViewerPose(data);
      }
      return this._pose;
    }
    getPose(space, baseSpace) {
      // Handle different space types
      if (
        space instanceof XRInputSourceSpace ||
        space instanceof XRJointSpace
      ) {
        if (space._pose) {
          return new XRPose(space._pose, false);
        }
      }
      if (space instanceof XRReferenceSpace) {
        return new XRPose(space._transform || new XRRigidTransform(), false);
      }
      return null;
    }
    getJointPose(joint, baseSpace) {
      if (joint instanceof XRJointSpace && joint._pose) {
        return new XRJointPose(joint._pose, joint.radius);
      }
      return null;
    }
  }

  // XRSubImage
  class XRSubImage {
    constructor(layer, view, session) {
      this._layer = layer;
      this._view = view;
      this._session = session;
      this._colorTextureCache = null;
      this._lastSwapchainIndex = null;
    }
    _invalidate() {
      this._colorTextureCache = null;
      this._lastSwapchainIndex = null;
    }
    get colorTexture() {
      const idx = this._session._currentSwapchainIndex;
      if (this._colorTextureCache && this._lastSwapchainIndex === idx)
        return this._colorTextureCache;
      this._lastSwapchainIndex = idx;
      const cachedView = this._layer._getTextureViewForEye(
        this._view._viewIndex
      );
      this._colorTextureCache = {
        _isXRSwapchain: true,
        width: this._layer._width,
        height: this._layer._height,
        depthOrArrayLayers: 1,
        createView: () => cachedView,
      };
      return this._colorTextureCache;
    }
    get depthStencilTexture() {
      return this._layer._depthTextures?.[this._view._viewIndex] ?? null;
    }
    get viewport() {
      return {
        x: 0,
        y: 0,
        width: this._layer._width,
        height: this._layer._height,
      };
    }
  }

  // XRProjectionLayer
  class XRProjectionLayer {
    constructor(session, device, init) {
      this._session = session;
      this._device = device;
      this._colorFormat = init.colorFormat || "rgba8unorm-srgb";
      this._depthStencilFormat = init.depthStencilFormat;
      this._width = session._info.width;
      this._height = session._info.height;
      this._viewCache = new Map();
      this._cachedSubImages = [null, null];
      this._lastSwapchainIndex = null;
      if (this._depthStencilFormat) {
        this._depthTextures = [
          device.createTexture({
            size: [this._width, this._height],
            format: this._depthStencilFormat,
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
          }),
          device.createTexture({
            size: [this._width, this._height],
            format: this._depthStencilFormat,
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
          }),
        ];
      }
    }
    _getTextureViewForEye(eyeIndex) {
      const idx = this._session._currentSwapchainIndex || 0;
      if (this._lastSwapchainIndex !== null && this._lastSwapchainIndex !== idx)
        this._cleanupSwapchainViews(this._lastSwapchainIndex);
      this._lastSwapchainIndex = idx;
      const key = `${idx}-${eyeIndex}`;
      if (!this._viewCache.has(key)) {
        const viewInfo =
          globalThis.__xr.op_xr_get_swapchain_texture_view(eyeIndex);
        this._viewCache.set(key, {
          __rid: viewInfo.rid,
          _isXRSwapchain: true,
          width: this._width,
          height: this._height,
        });
      }
      return this._viewCache.get(key);
    }
    _cleanupSwapchainViews(oldIdx) {
      for (let eye = 0; eye < 2; eye++) {
        const key = `${oldIdx}-${eye}`;
        const view = this._viewCache.get(key);
        if (view?.__rid != null)
          try {
            globalThis.__gfx?.op_gfx_resource_drop?.(view.__rid);
          } catch {}
        this._viewCache.delete(key);
      }
    }
    _invalidateCache() {
      for (const [, v] of this._viewCache)
        if (v?.__rid != null)
          try {
            globalThis.__gfx?.op_gfx_resource_drop?.(v.__rid);
          } catch {}
      this._viewCache.clear();
      this._cachedSubImages = [null, null];
      this._lastSwapchainIndex = null;
    }
    get textureWidth() {
      return this._width;
    }
    get textureHeight() {
      return this._height;
    }
    get ignoreDepthValues() {
      return false;
    }
    get fixedFoveation() {
      return 0;
    }
    set fixedFoveation(v) {}
  }

  // XRWebGLLayer
  class XRWebGLLayer {
    constructor(session, context, init = {}) {
      this._session = session;
      this._context = context;
      this._antialias = init.antialias ?? true;
      this._depth = init.depth ?? true;
      this._stencil = init.stencil ?? false;
      this._alpha = init.alpha ?? true;
      this._ignoreDepthValues = init.ignoreDepthValues ?? false;
      this._framebufferScaleFactor = init.framebufferScaleFactor ?? 1.0;
      this._framebufferWidth = session._info.width * 2;
      this._framebufferHeight = session._info.height;
    }
    get antialias() {
      return this._antialias;
    }
    get ignoreDepthValues() {
      return this._ignoreDepthValues;
    }
    get framebufferWidth() {
      return this._framebufferWidth;
    }
    get framebufferHeight() {
      return this._framebufferHeight;
    }
    get framebuffer() {
      return null;
    }
    getViewport(view) {
      return {
        x: view._viewIndex * this._session._info.width,
        y: 0,
        width: this._session._info.width,
        height: this._session._info.height,
      };
    }
    static getNativeFramebufferScaleFactor() {
      return 1.0;
    }
  }

  // XRGPUBinding
  class XRGPUBinding {
    constructor(session, device) {
      this._session = session;
      this._device = device;
      this._projectionLayer = null;
    }
    createProjectionLayer(init = {}) {
      this._projectionLayer = new XRProjectionLayer(
        this._session,
        this._device,
        init
      );
      return this._projectionLayer;
    }
    getViewSubImage(layer, view) {
      const key = `${layer}_${view._viewIndex}`;
      const frame = globalThis.__currentXRFrame;
      if (frame?._viewSubImageCache.has(key))
        return frame._viewSubImageCache.get(key);
      if (!layer._cachedSubImages[view._viewIndex]) {
        layer._cachedSubImages[view._viewIndex] = new XRSubImage(
          layer,
          view,
          this._session
        );
      }
      const subImage = layer._cachedSubImages[view._viewIndex];
      if (frame) frame._viewSubImageCache.set(key, subImage);
      return subImage;
    }
    getPreferredColorFormat() {
      return "rgba8unorm-srgb";
    }
  }

  // XRSession
  class XRSession extends SimpleEventTarget {
    constructor(info, mode, options) {
      super();
      this._info = info;
      this._mode = mode;
      this._options = options;
      this._ended = false;
      this._rafCallbacks = [];
      this._rafId = 0;
      this._loopRunning = false;
      this._referenceSpaces = new Map();
      this._currentSwapchainIndex = null;
      this._renderState = {
        baseLayer: null,
        depthNear: 0.1,
        depthFar: 1000.0,
        inlineVerticalFieldOfView: null,
        layers: [],
      };
      this._inputSources = new XRInputSourceArray();
      this._visibilityState = "visible";
      globalThis.__xrMode = true;
    }
    get renderState() {
      return this._renderState;
    }
    get inputSources() {
      return this._inputSources;
    }
    get visibilityState() {
      return this._visibilityState;
    }
    get environmentBlendMode() {
      return this._mode === "immersive-ar" ? "alpha-blend" : "opaque";
    }

    updateRenderState(state) {
      if (state.baseLayer !== undefined)
        this._renderState.baseLayer = state.baseLayer;
      if (state.depthNear !== undefined)
        this._renderState.depthNear = state.depthNear;
      if (state.depthFar !== undefined)
        this._renderState.depthFar = state.depthFar;
      if (state.inlineVerticalFieldOfView !== undefined)
        this._renderState.inlineVerticalFieldOfView =
          state.inlineVerticalFieldOfView;
      if (state.layers !== undefined) this._renderState.layers = state.layers;
    }

    async requestReferenceSpace(type) {
      if (this._referenceSpaces.has(type))
        return this._referenceSpaces.get(type);
      const space =
        type === "bounded-floor"
          ? new XRBoundedReferenceSpace()
          : new XRReferenceSpace(type);
      this._referenceSpaces.set(type, space);
      return space;
    }

    requestAnimationFrame(callback) {
      const id = ++this._rafId;
      this._rafCallbacks.push({ id, callback });
      if (!this._loopRunning) this._startNativeLoop();
      return id;
    }

    cancelAnimationFrame(id) {
      this._rafCallbacks = this._rafCallbacks.filter((c) => c.id !== id);
    }

    async _startNativeLoop() {
      if (this._loopRunning) return;
      this._loopRunning = true;

      while (this._loopRunning && !this._ended) {
        try {
          if (!globalThis.__xr.op_xr_poll_events()) {
            this.end();
            return;
          }
          const frameState = await globalThis.__xr.op_xr_wait_frame();
          if (frameState.should_render) {
            const poseData = globalThis.__xr.op_xr_get_viewer_pose();
            const swapchainInfo =
              globalThis.__xr.op_xr_acquire_swapchain_image();
            this._currentSwapchainIndex = swapchainInfo.index;

            // Update input sources from native
            let hasNativeInput = false;
            try {
              const inputState = globalThis.__xr.op_xr_get_input_sources();
              if (
                inputState &&
                inputState.sources &&
                inputState.sources.length > 0
              ) {
                this._inputSources._updateFromNative(inputState.sources);
                hasNativeInput = true;
              }
            } catch (e) {
              // Native input not available
            }

            // If no native input, create synthetic input sources so controllers work
            if (!hasNativeInput && this._inputSources.length === 0) {
              // Create left and right controller sources with default poses
              const defaultSources = [
                {
                  handedness: "left",
                  target_ray_mode: "tracked-pointer",
                  profiles: ["generic-trigger-squeeze-thumbstick"],
                  grip_space_pose: {
                    position: [-0.2, 1.0, -0.3],
                    orientation: [0, 0, 0, 1],
                    matrix: [
                      1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, -0.2, 1.0, -0.3, 1,
                    ],
                  },
                  target_ray_pose: {
                    position: [-0.2, 1.0, -0.3],
                    orientation: [0, 0, 0, 1],
                    matrix: [
                      1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, -0.2, 1.0, -0.3, 1,
                    ],
                  },
                  gamepad: {
                    buttons: [
                      { pressed: false, touched: false, value: 0 },
                      { pressed: false, touched: false, value: 0 },
                      { pressed: false, touched: false, value: 0 },
                      { pressed: false, touched: false, value: 0 },
                      { pressed: false, touched: false, value: 0 },
                      { pressed: false, touched: false, value: 0 },
                    ],
                    axes: [0, 0, 0, 0],
                  },
                  hand: null,
                },
                {
                  handedness: "right",
                  target_ray_mode: "tracked-pointer",
                  profiles: ["generic-trigger-squeeze-thumbstick"],
                  grip_space_pose: {
                    position: [0.2, 1.0, -0.3],
                    orientation: [0, 0, 0, 1],
                    matrix: [
                      1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0.2, 1.0, -0.3, 1,
                    ],
                  },
                  target_ray_pose: {
                    position: [0.2, 1.0, -0.3],
                    orientation: [0, 0, 0, 1],
                    matrix: [
                      1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0.2, 1.0, -0.3, 1,
                    ],
                  },
                  gamepad: {
                    buttons: [
                      { pressed: false, touched: false, value: 0 },
                      { pressed: false, touched: false, value: 0 },
                      { pressed: false, touched: false, value: 0 },
                      { pressed: false, touched: false, value: 0 },
                      { pressed: false, touched: false, value: 0 },
                      { pressed: false, touched: false, value: 0 },
                    ],
                    axes: [0, 0, 0, 0],
                  },
                  hand: null,
                },
              ];
              this._inputSources._updateFromNative(defaultSources);
            }

            // Note: Connected events are fired by XR.js updateXRInputSources, not here

            const xrFrame = new XRFrame(this, frameState);
            xrFrame._cachedPose = poseData;
            globalThis.__currentXRFrame = xrFrame;

            const callbacks = this._rafCallbacks;
            this._rafCallbacks = [];
            const time = frameState.predicted_display_time / 1_000_000;

            for (const { callback } of callbacks) {
              try {
                callback(time, xrFrame);
              } catch (e) {
                console.error("XR frame error:", e);
              }
            }

            globalThis.__currentXRFrame = null;
            globalThis.__xr.op_xr_release_swapchain_image();
            globalThis.__xr.op_xr_end_frame();
          }
        } catch (e) {
          console.error("XR loop error:", e);
          this.end();
          return;
        }
      }
    }

    async end() {
      if (this._ended) return;
      this._ended = true;
      this._loopRunning = false;
      globalThis.__xrMode = false;
      for (const layer of this._renderState.layers || [])
        layer._invalidateCache?.();
      try {
        globalThis.__xr.op_xr_end_session();
      } catch {}
      this.dispatchEvent({
        type: "end",
        session: this,
        bubbles: false,
        cancelable: false,
        timeStamp: performance.now(),
      });
    }
  }

  // XRSystem
  class XRSystem extends SimpleEventTarget {
    async isSessionSupported(mode) {
      if (mode !== "immersive-vr" && mode !== "immersive-ar") return false;
      try {
        return await globalThis.__xr.op_xr_is_supported();
      } catch {
        return false;
      }
    }

    async requestSession(mode, options = {}) {
      if (__xrSession && !__xrSession._ended) return __xrSession;
      const sessionInfo = globalThis.__xrSessionInfo;
      if (!sessionInfo)
        throw new Error(
          "XR session not initialized - call navigator.gpu.requestAdapter first"
        );
      __xrSession = new XRSession(sessionInfo, mode, options);
      return __xrSession;
    }
  }

  // Install
  navigator.xr = new XRSystem();
  globalThis.XRSystem = XRSystem;
  globalThis.XRSession = XRSession;
  globalThis.XRFrame = XRFrame;
  globalThis.XRView = XRView;
  globalThis.XRViewerPose = XRViewerPose;
  globalThis.XRPose = XRPose;
  globalThis.XRJointPose = XRJointPose;
  globalThis.XRRigidTransform = XRRigidTransform;
  globalThis.XRReferenceSpace = XRReferenceSpace;
  globalThis.XRBoundedReferenceSpace = XRBoundedReferenceSpace;
  globalThis.XRSpace = XRSpace;
  globalThis.XRInputSource = XRInputSource;
  globalThis.XRInputSourceArray = XRInputSourceArray;
  globalThis.XRHand = XRHand;
  globalThis.XRJointSpace = XRJointSpace;
  globalThis.XRWebGLLayer = XRWebGLLayer;
  globalThis.XRGPUBinding = XRGPUBinding;
  globalThis.XRProjectionLayer = XRProjectionLayer;
  globalThis.XRSubImage = XRSubImage;

  // Helpers
  globalThis.initNativeXR = async () => {
    if (globalThis.__xrSessionInfo) return globalThis.__xrSessionInfo;
    globalThis.__xrSessionInfo = await globalThis.__xr.op_xr_request_session();
    return globalThis.__xrSessionInfo;
  };
  globalThis.getNativeXRSessionInfo = () => globalThis.__xrSessionInfo;
  globalThis.isNativeXR = () => true;

  console.log("[shims] WebXR initialized with controller support");
}
