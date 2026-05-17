// 02-input.js - Input event handling

function dispatchToTargets(evt) {
  for (const canvas of globalThis.__inputCanvases || []) {
    canvas.dispatchEvent?.(evt);
  }
  globalThis.document?.body?.dispatchEvent?.(evt);
  globalThis.document?.dispatchEvent?.(evt);
  globalThis.window?.dispatchEvent?.(evt);
}

function synthesizeMouseFromPointer(pEvt) {
  if (pEvt.pointerType && pEvt.pointerType !== "mouse") return;

  const typeMap = {
    pointermove: "mousemove",
    pointerdown: "mousedown",
    pointerup: "mouseup",
    contextmenu: "contextmenu",
  };
  const type = typeMap[pEvt.type];
  if (!type) return;

  dispatchToTargets({
    ...pEvt,
    type,
    which: pEvt.buttons
      ? pEvt.buttons & 1
        ? 1
        : pEvt.buttons & 4
        ? 2
        : pEvt.buttons & 2
        ? 3
        : 0
      : pEvt.button === 0
      ? 1
      : pEvt.button === 1
      ? 2
      : pEvt.button === 2
      ? 3
      : 0,
  });
}

function synthesizeLegacyWheel(wEvt) {
  dispatchToTargets({
    ...wEvt,
    type: "mousewheel",
    wheelDelta: -wEvt.deltaY * 3,
  });
  dispatchToTargets({ ...wEvt, type: "DOMMouseScroll", detail: wEvt.deltaY });
}

function createDOMEvent(evt) {
  return {
    ...evt,
    bubbles: true,
    cancelable: true,
    defaultPrevented: false,
    propagationStopped: false,
    immediatePropagationStopped: false,
    timeStamp: performance.now(),
    isTrusted: true,
    target: null,
    currentTarget: null,
    preventDefault() {
      this.defaultPrevented = true;
    },
    stopPropagation() {
      this.propagationStopped = true;
    },
    stopImmediatePropagation() {
      this.propagationStopped = true;
      this.immediatePropagationStopped = true;
    },
    getModifierState(key) {
      const map = {
        Control: "ctrlKey",
        Shift: "shiftKey",
        Alt: "altKey",
        Meta: "metaKey",
      };
      return !!this[map[key]];
    },
    pageX: evt.clientX,
    pageY: evt.clientY,
  };
}

globalThis.__dispatchInputEvents = function () {
  if (!globalThis.__input?.op_input_poll_events) return;

  try {
    for (const evt of globalThis.__input.op_input_poll_events()) {
      const domEvent = createDOMEvent(evt);
      const { type } = domEvent;

      // Pointer lock change
      if (type === "pointerlockchange") {
        if (!domEvent.locked) globalThis.__setPointerLockElement?.(null);
        globalThis.document?.dispatchEvent?.(domEvent);
        continue;
      }

      // Pointer events
      if (
        [
          "pointerdown",
          "pointerup",
          "pointermove",
          "pointercancel",
          "contextmenu",
        ].includes(type)
      ) {
        dispatchToTargets(domEvent);
        synthesizeMouseFromPointer(domEvent);
        continue;
      }

      // Wheel
      if (type === "wheel") {
        dispatchToTargets(domEvent);
        synthesizeLegacyWheel(domEvent);
        continue;
      }

      // Keyboard
      if (type === "keydown" || type === "keyup") {
        // Skip repeat keydowns — in the browser, timing naturally spaces
        // repeated events across frames. In our batched dispatch, repeats
        // can retrigger actions before the previous action's release phase
        // completes, causing state corruption in audio schedulers.
        if (type === "keydown" && domEvent.repeat) continue;
        if (type === "keydown" && domEvent.key === "Escape") {
          if (globalThis.__input.op_input_is_pointer_locked?.()) {
            globalThis.__input.op_input_exit_pointer_lock?.();
          }
          if (globalThis.__getPointerLockElement?.()) {
            globalThis.document?.exitPointerLock?.();
          }
        }
        globalThis.document?.dispatchEvent?.(domEvent);
        globalThis.window?.dispatchEvent?.(domEvent);
        continue;
      }

      // Resize/focus/blur
      if (type === "resize" || type === "focus" || type === "blur") {
        if (type === "resize" && domEvent.width && domEvent.height) {
          globalThis.__windowInnerWidth = domEvent.width;
          globalThis.__windowInnerHeight = domEvent.height;
          if (globalThis.window) {
            globalThis.window.innerWidth = domEvent.width;
            globalThis.window.innerHeight = domEvent.height;
          }
          for (const canvas of globalThis.__inputCanvases || []) {
            canvas.width = domEvent.width;
            canvas.height = domEvent.height;
          }
        }
        globalThis.window?.dispatchEvent?.(domEvent);
        globalThis.document?.dispatchEvent?.(domEvent);
      }
    }
  } catch (e) {
    console.error("Input dispatch error:", e);
  }
};

console.log("[shims] input initialized");
