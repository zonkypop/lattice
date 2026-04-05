// src/input.rs
//
// User input handling: captures winit events and exposes them to JavaScript
// via ops that can be polled or via callbacks.

use deno_core::op2;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex, OnceLock};

#[cfg(not(target_os = "android"))]
use winit::event::{ElementState, MouseButton, MouseScrollDelta, TouchPhase};
#[cfg(not(target_os = "android"))]
use winit::keyboard::{Key, KeyCode, ModifiersState, NamedKey, PhysicalKey};

// ======================= Global Event Queue =======================

static INPUT_QUEUE: OnceLock<Arc<Mutex<InputEventQueue>>> = OnceLock::new();

pub fn get_input_queue() -> Arc<Mutex<InputEventQueue>> {
    INPUT_QUEUE
        .get_or_init(|| Arc::new(Mutex::new(InputEventQueue::default())))
        .clone()
}

/// Global event queue that stores input events from winit
#[derive(Default)]
pub struct InputEventQueue {
    events: VecDeque<InputEvent>,
    pointers: std::collections::HashMap<i32, PointerState>,
    modifiers: ModifierState,
    pub window_width: u32,
    pub window_height: u32,
    pub pointer_lock_requested: bool,
    pub pointer_lock_exit_requested: bool,
    pub pointer_locked: bool,
    pub cursor_style_requested: Option<String>,
}

#[derive(Clone, Default)]
struct PointerState {
    x: f64,
    y: f64,
    buttons: u16,
}

#[derive(Clone, Default)]
struct ModifierState {
    ctrl: bool,
    shift: bool,
    alt: bool,
    meta: bool,
}

impl InputEventQueue {
    pub fn set_window_size(&mut self, width: u32, height: u32) {
        self.window_width = width;
        self.window_height = height;
    }

    pub fn handle_raw_mouse_motion(&mut self, delta_x: f64, delta_y: f64) {
        let pointer_id = 0;
        
        let (x, y) = self
            .pointers
            .get(&pointer_id)
            .map(|p| (p.x, p.y))
            .unwrap_or((0.0, 0.0));

        let buttons = self
            .pointers
            .get(&pointer_id)
            .map(|p| p.buttons)
            .unwrap_or(0);

        let event = InputEvent::PointerMove(PointerEventData {
            pointer_id,
            pointer_type: "mouse".to_string(),
            button: -1,
            buttons,
            client_x: x,
            client_y: y,
            offset_x: x,
            offset_y: y,
            screen_x: x,
            screen_y: y,
            movement_x: delta_x,
            movement_y: delta_y,
            ctrl_key: self.modifiers.ctrl,
            shift_key: self.modifiers.shift,
            alt_key: self.modifiers.alt,
            meta_key: self.modifiers.meta,
            is_primary: true,
            pressure: if buttons > 0 { 0.5 } else { 0.0 },
            width: 1.0,
            height: 1.0,
            tilt_x: 0,
            tilt_y: 0,
            twist: 0,
        });

        self.push_event(event);
    }

    pub fn push_event(&mut self, event: InputEvent) {
        self.events.push_back(event);
    }

    pub fn drain_events(&mut self) -> Vec<InputEvent> {
        self.events.drain(..).collect()
    }

    #[cfg(not(target_os = "android"))]
    pub fn set_modifiers(&mut self, state: ModifiersState) {
        self.modifiers.ctrl = state.control_key();
        self.modifiers.shift = state.shift_key();
        self.modifiers.alt = state.alt_key();
        self.modifiers.meta = state.super_key();
    }

    pub fn update_pointer(&mut self, pointer_id: i32, x: f64, y: f64, buttons: Option<u16>) {
        let state = self.pointers.entry(pointer_id).or_default();
        state.x = x;
        state.y = y;
        if let Some(b) = buttons {
            state.buttons = b;
        }
    }

    pub fn remove_pointer(&mut self, pointer_id: i32) {
        self.pointers.remove(&pointer_id);
    }
}

// ======================= Event Types =======================

#[derive(Clone, Serialize, Deserialize, Debug)]
#[serde(tag = "type")]
pub enum InputEvent {
    #[serde(rename = "pointerdown")]
    PointerDown(PointerEventData),
    #[serde(rename = "pointerup")]
    PointerUp(PointerEventData),
    #[serde(rename = "pointermove")]
    PointerMove(PointerEventData),
    #[serde(rename = "pointercancel")]
    PointerCancel(PointerEventData),
    #[serde(rename = "wheel")]
    Wheel(WheelEventData),
    #[serde(rename = "keydown")]
    KeyDown(KeyEventData),
    #[serde(rename = "keyup")]
    KeyUp(KeyEventData),
    #[serde(rename = "contextmenu")]
    ContextMenu(PointerEventData),
    #[serde(rename = "resize")]
    Resize(ResizeEventData),
    #[serde(rename = "focus")]
    Focus,
    #[serde(rename = "blur")]
    Blur,
    #[serde(rename = "pointerlockchange")]
    PointerLockChange { locked: bool },
}

#[derive(Clone, Serialize, Deserialize, Debug, Default)]
pub struct PointerEventData {
    #[serde(rename = "pointerId")]
    pub pointer_id: i32,
    #[serde(rename = "pointerType")]
    pub pointer_type: String, // "mouse", "touch", "pen"
    pub button: i16,          // -1 for move, 0=left, 1=middle, 2=right
    pub buttons: u16,         // bitmask of pressed buttons
    #[serde(rename = "clientX")]
    pub client_x: f64,
    #[serde(rename = "clientY")]
    pub client_y: f64,
    #[serde(rename = "offsetX")]
    pub offset_x: f64,
    #[serde(rename = "offsetY")]
    pub offset_y: f64,
    #[serde(rename = "screenX")]
    pub screen_x: f64,
    #[serde(rename = "screenY")]
    pub screen_y: f64,
    #[serde(rename = "movementX")]
    pub movement_x: f64,
    #[serde(rename = "movementY")]
    pub movement_y: f64,
    #[serde(rename = "ctrlKey")]
    pub ctrl_key: bool,
    #[serde(rename = "shiftKey")]
    pub shift_key: bool,
    #[serde(rename = "altKey")]
    pub alt_key: bool,
    #[serde(rename = "metaKey")]
    pub meta_key: bool,
    #[serde(rename = "isPrimary")]
    pub is_primary: bool,
    pub pressure: f32,
    pub width: f32,
    pub height: f32,
    #[serde(rename = "tiltX")]
    pub tilt_x: i32,
    #[serde(rename = "tiltY")]
    pub tilt_y: i32,
    pub twist: i32,
}

#[derive(Clone, Serialize, Deserialize, Debug, Default)]
pub struct WheelEventData {
    #[serde(rename = "deltaX")]
    pub delta_x: f64,
    #[serde(rename = "deltaY")]
    pub delta_y: f64,
    #[serde(rename = "deltaZ")]
    pub delta_z: f64,
    #[serde(rename = "deltaMode")]
    pub delta_mode: u32, // 0=pixel, 1=line, 2=page
    #[serde(rename = "clientX")]
    pub client_x: f64,
    #[serde(rename = "clientY")]
    pub client_y: f64,
    #[serde(rename = "ctrlKey")]
    pub ctrl_key: bool,
    #[serde(rename = "shiftKey")]
    pub shift_key: bool,
    #[serde(rename = "altKey")]
    pub alt_key: bool,
    #[serde(rename = "metaKey")]
    pub meta_key: bool,
}

#[derive(Clone, Serialize, Deserialize, Debug, Default)]
pub struct KeyEventData {
    pub key: String,   // "a", "Enter", "ArrowUp", etc.
    pub code: String,  // "KeyA", "Enter", "ArrowUp", etc.
    #[serde(rename = "keyCode")]
    pub key_code: u32, // legacy keyCode
    pub location: u32, // 0=standard, 1=left, 2=right, 3=numpad
    pub repeat: bool,
    #[serde(rename = "ctrlKey")]
    pub ctrl_key: bool,
    #[serde(rename = "shiftKey")]
    pub shift_key: bool,
    #[serde(rename = "altKey")]
    pub alt_key: bool,
    #[serde(rename = "metaKey")]
    pub meta_key: bool,
}

#[derive(Clone, Serialize, Deserialize, Debug, Default)]
pub struct ResizeEventData {
    pub width: u32,
    pub height: u32,
}

// ======================= Winit Event Conversion =======================

impl InputEventQueue {
    /// Handle a winit CursorMoved event
    pub fn handle_cursor_moved(&mut self, x: f64, y: f64) {
        let pointer_id = 0; // Mouse is always pointer 0

        let (movement_x, movement_y) = if let Some(prev) = self.pointers.get(&pointer_id) {
            (x - prev.x, y - prev.y)
        } else {
            (0.0, 0.0)
        };

        let buttons = self
            .pointers
            .get(&pointer_id)
            .map(|p| p.buttons)
            .unwrap_or(0);
        self.update_pointer(pointer_id, x, y, None);

        let event = InputEvent::PointerMove(PointerEventData {
            pointer_id,
            pointer_type: "mouse".to_string(),
            button: -1,
            buttons,
            client_x: x,
            client_y: y,
            offset_x: x,
            offset_y: y,
            screen_x: x,
            screen_y: y,
            movement_x,
            movement_y,
            ctrl_key: self.modifiers.ctrl,
            shift_key: self.modifiers.shift,
            alt_key: self.modifiers.alt,
            meta_key: self.modifiers.meta,
            is_primary: true,
            pressure: if buttons > 0 { 0.5 } else { 0.0 },
            width: 1.0,
            height: 1.0,
            tilt_x: 0,
            tilt_y: 0,
            twist: 0,
        });

        self.push_event(event);
    }

    /// Handle a winit MouseInput event
    #[cfg(not(target_os = "android"))]
    pub fn handle_mouse_input(&mut self, state: ElementState, button: MouseButton) {
        let pointer_id = 0;

        let (button_num, button_mask): (i16, u16) = match button {
            MouseButton::Left => (0, 1),
            MouseButton::Middle => (1, 4),
            MouseButton::Right => (2, 2),
            MouseButton::Back => (3, 8),
            MouseButton::Forward => (4, 16),
            MouseButton::Other(n) => (n as i16, 1 << (n.min(15))),
        };

        let (x, y) = self
            .pointers
            .get(&pointer_id)
            .map(|p| (p.x, p.y))
            .unwrap_or((0.0, 0.0));

        let buttons = {
            let current = self
                .pointers
                .get(&pointer_id)
                .map(|p| p.buttons)
                .unwrap_or(0);
            match state {
                ElementState::Pressed => current | button_mask,
                ElementState::Released => current & !button_mask,
            }
        };

        self.update_pointer(pointer_id, x, y, Some(buttons));

        let event_data = PointerEventData {
            pointer_id,
            pointer_type: "mouse".to_string(),
            button: button_num,
            buttons,
            client_x: x,
            client_y: y,
            offset_x: x,
            offset_y: y,
            screen_x: x,
            screen_y: y,
            movement_x: 0.0,
            movement_y: 0.0,
            ctrl_key: self.modifiers.ctrl,
            shift_key: self.modifiers.shift,
            alt_key: self.modifiers.alt,
            meta_key: self.modifiers.meta,
            is_primary: true,
            pressure: if state == ElementState::Pressed {
                0.5
            } else {
                0.0
            },
            width: 1.0,
            height: 1.0,
            tilt_x: 0,
            tilt_y: 0,
            twist: 0,
        };

        let event = match state {
            ElementState::Pressed => InputEvent::PointerDown(event_data),
            ElementState::Released => InputEvent::PointerUp(event_data),
        };

        self.push_event(event);

        // Also emit contextmenu for right-click
        if button == MouseButton::Right && state == ElementState::Released {
            self.push_event(InputEvent::ContextMenu(PointerEventData {
                pointer_id,
                pointer_type: "mouse".to_string(),
                button: 2,
                buttons,
                client_x: x,
                client_y: y,
                offset_x: x,
                offset_y: y,
                screen_x: x,
                screen_y: y,
                movement_x: 0.0,
                movement_y: 0.0,
                ctrl_key: self.modifiers.ctrl,
                shift_key: self.modifiers.shift,
                alt_key: self.modifiers.alt,
                meta_key: self.modifiers.meta,
                is_primary: true,
                pressure: 0.0,
                width: 1.0,
                height: 1.0,
                tilt_x: 0,
                tilt_y: 0,
                twist: 0,
            }));
        }
    }

    /// Handle a winit MouseWheel event
    #[cfg(not(target_os = "android"))]
    pub fn handle_mouse_wheel(&mut self, delta: MouseScrollDelta, _phase: TouchPhase) {
        let pointer_id = 0;
        let (x, y) = self
            .pointers
            .get(&pointer_id)
            .map(|p| (p.x, p.y))
            .unwrap_or((0.0, 0.0));

        let (delta_x, delta_y, delta_mode) = match delta {
            MouseScrollDelta::LineDelta(x, y) => {
                // Line-based scrolling (typical for mouse wheel)
                // 100 px approximately what chrome and fire use
                (x as f64 * 100.0, y as f64 * 100.0, 0u32)
            }
            MouseScrollDelta::PixelDelta(pos) => (pos.x, pos.y, 0u32),
        };

        let event = InputEvent::Wheel(WheelEventData {
            delta_x,
            delta_y: -delta_y, // Invert Y to match browser behavior
            delta_z: 0.0,
            delta_mode,
            client_x: x,
            client_y: y,
            ctrl_key: self.modifiers.ctrl,
            shift_key: self.modifiers.shift,
            alt_key: self.modifiers.alt,
            meta_key: self.modifiers.meta,
        });

        self.push_event(event);
    }

    /// Handle a winit KeyboardInput event
    #[cfg(not(target_os = "android"))]
    pub fn handle_keyboard_input(
        &mut self,
        state: ElementState,
        physical_key: PhysicalKey,
        logical_key: &Key,
        repeat: bool,
    ) {
        let (_fallback_key, code, key_code, location) = match physical_key {
            PhysicalKey::Code(kc) => keycode_to_js(kc),
            PhysicalKey::Unidentified(_) => {
                ("Unidentified".to_string(), "Unidentified".to_string(), 0, 0)
            }
        };

        // Use logical key for the `key` field to respect keyboard layout and modifiers
        // (e.g. Shift+2 → "@" on US layout, Shift+a → "A")
        let key = logical_key_to_js_key(logical_key);

        let event_data = KeyEventData {
            key,
            code,
            key_code,
            location,
            repeat,
            ctrl_key: self.modifiers.ctrl,
            shift_key: self.modifiers.shift,
            alt_key: self.modifiers.alt,
            meta_key: self.modifiers.meta,
        };

        let event = match state {
            ElementState::Pressed => InputEvent::KeyDown(event_data),
            ElementState::Released => InputEvent::KeyUp(event_data),
        };

        self.push_event(event);
    }

    /// Handle a winit Touch event
    #[cfg(not(target_os = "android"))]
    pub fn handle_touch(&mut self, touch: winit::event::Touch) {
        let pointer_id = touch.id as i32;
        let x = touch.location.x;
        let y = touch.location.y;

        let (movement_x, movement_y) = if let Some(prev) = self.pointers.get(&pointer_id) {
            (x - prev.x, y - prev.y)
        } else {
            (0.0, 0.0)
        };

        let is_primary = pointer_id == 0;

        let event_data = PointerEventData {
            pointer_id,
            pointer_type: "touch".to_string(),
            button: 0,
            buttons: match touch.phase {
                TouchPhase::Started | TouchPhase::Moved => 1,
                _ => 0,
            },
            client_x: x,
            client_y: y,
            offset_x: x,
            offset_y: y,
            screen_x: x,
            screen_y: y,
            movement_x,
            movement_y,
            ctrl_key: self.modifiers.ctrl,
            shift_key: self.modifiers.shift,
            alt_key: self.modifiers.alt,
            meta_key: self.modifiers.meta,
            is_primary,
            pressure: touch
                .force
                .map(|f| match f {
                    winit::event::Force::Normalized(n) => n as f32,
                    winit::event::Force::Calibrated {
                        force,
                        max_possible_force,
                        ..
                    } => (force / max_possible_force) as f32,
                })
                .unwrap_or(0.5),
            width: 1.0,
            height: 1.0,
            tilt_x: 0,
            tilt_y: 0,
            twist: 0,
        };

        let event = match touch.phase {
            TouchPhase::Started => {
                self.update_pointer(pointer_id, x, y, Some(1));
                InputEvent::PointerDown(event_data)
            }
            TouchPhase::Moved => {
                self.update_pointer(pointer_id, x, y, None);
                InputEvent::PointerMove(event_data)
            }
            TouchPhase::Ended => {
                self.remove_pointer(pointer_id);
                InputEvent::PointerUp(event_data)
            }
            TouchPhase::Cancelled => {
                self.remove_pointer(pointer_id);
                InputEvent::PointerCancel(event_data)
            }
        };

        self.push_event(event);
    }

    /// Handle window resize
    pub fn handle_resize(&mut self, width: u32, height: u32) {
        self.window_width = width;
        self.window_height = height;
        self.push_event(InputEvent::Resize(ResizeEventData { width, height }));
    }

    /// Handle window focus
    pub fn handle_focus(&mut self, focused: bool) {
        if focused {
            self.push_event(InputEvent::Focus);
        } else {
            self.push_event(InputEvent::Blur);
        }
    }
}

// ======================= Logical Key Mapping =======================

/// Convert winit's logical Key to the browser KeyboardEvent.key value.
/// This respects keyboard layout and modifier state (e.g. Shift+2 → "@").
#[cfg(not(target_os = "android"))]
fn logical_key_to_js_key(logical_key: &Key) -> String {
    match logical_key {
        Key::Character(s) => s.to_string(),
        Key::Named(named) => named_key_to_js_string(named),
        Key::Dead(_) => "Dead".to_string(),
        Key::Unidentified(_) => "Unidentified".to_string(),
    }
}

#[cfg(not(target_os = "android"))]
fn named_key_to_js_string(named: &NamedKey) -> String {
    use NamedKey::*;
    match named {
        Shift => "Shift",
        Control => "Control",
        Alt => "Alt",
        Super => "Meta",
        Enter => "Enter",
        Tab => "Tab",
        Space => " ",
        Backspace => "Backspace",
        Escape => "Escape",
        Delete => "Delete",
        Insert => "Insert",
        Home => "Home",
        End => "End",
        PageUp => "PageUp",
        PageDown => "PageDown",
        CapsLock => "CapsLock",
        ArrowUp => "ArrowUp",
        ArrowDown => "ArrowDown",
        ArrowLeft => "ArrowLeft",
        ArrowRight => "ArrowRight",
        F1 => "F1",
        F2 => "F2",
        F3 => "F3",
        F4 => "F4",
        F5 => "F5",
        F6 => "F6",
        F7 => "F7",
        F8 => "F8",
        F9 => "F9",
        F10 => "F10",
        F11 => "F11",
        F12 => "F12",
        NumLock => "NumLock",
        ScrollLock => "ScrollLock",
        PrintScreen => "PrintScreen",
        Pause => "Pause",
        ContextMenu => "ContextMenu",
        _ => "Unidentified",
    }
    .to_string()
}

// ======================= Keycode Mapping =======================

#[cfg(not(target_os = "android"))]
fn keycode_to_js(keycode: KeyCode) -> (String, String, u32, u32) {
    // Returns (key, code, keyCode, location)
    // location: 0=standard, 1=left, 2=right, 3=numpad
    use KeyCode::*;

    match keycode {
        // Letters
        KeyA => ("a".into(), "KeyA".into(), 65, 0),
        KeyB => ("b".into(), "KeyB".into(), 66, 0),
        KeyC => ("c".into(), "KeyC".into(), 67, 0),
        KeyD => ("d".into(), "KeyD".into(), 68, 0),
        KeyE => ("e".into(), "KeyE".into(), 69, 0),
        KeyF => ("f".into(), "KeyF".into(), 70, 0),
        KeyG => ("g".into(), "KeyG".into(), 71, 0),
        KeyH => ("h".into(), "KeyH".into(), 72, 0),
        KeyI => ("i".into(), "KeyI".into(), 73, 0),
        KeyJ => ("j".into(), "KeyJ".into(), 74, 0),
        KeyK => ("k".into(), "KeyK".into(), 75, 0),
        KeyL => ("l".into(), "KeyL".into(), 76, 0),
        KeyM => ("m".into(), "KeyM".into(), 77, 0),
        KeyN => ("n".into(), "KeyN".into(), 78, 0),
        KeyO => ("o".into(), "KeyO".into(), 79, 0),
        KeyP => ("p".into(), "KeyP".into(), 80, 0),
        KeyQ => ("q".into(), "KeyQ".into(), 81, 0),
        KeyR => ("r".into(), "KeyR".into(), 82, 0),
        KeyS => ("s".into(), "KeyS".into(), 83, 0),
        KeyT => ("t".into(), "KeyT".into(), 84, 0),
        KeyU => ("u".into(), "KeyU".into(), 85, 0),
        KeyV => ("v".into(), "KeyV".into(), 86, 0),
        KeyW => ("w".into(), "KeyW".into(), 87, 0),
        KeyX => ("x".into(), "KeyX".into(), 88, 0),
        KeyY => ("y".into(), "KeyY".into(), 89, 0),
        KeyZ => ("z".into(), "KeyZ".into(), 90, 0),

        // Numbers
        Digit0 => ("0".into(), "Digit0".into(), 48, 0),
        Digit1 => ("1".into(), "Digit1".into(), 49, 0),
        Digit2 => ("2".into(), "Digit2".into(), 50, 0),
        Digit3 => ("3".into(), "Digit3".into(), 51, 0),
        Digit4 => ("4".into(), "Digit4".into(), 52, 0),
        Digit5 => ("5".into(), "Digit5".into(), 53, 0),
        Digit6 => ("6".into(), "Digit6".into(), 54, 0),
        Digit7 => ("7".into(), "Digit7".into(), 55, 0),
        Digit8 => ("8".into(), "Digit8".into(), 56, 0),
        Digit9 => ("9".into(), "Digit9".into(), 57, 0),

        // Function keys
        F1 => ("F1".into(), "F1".into(), 112, 0),
        F2 => ("F2".into(), "F2".into(), 113, 0),
        F3 => ("F3".into(), "F3".into(), 114, 0),
        F4 => ("F4".into(), "F4".into(), 115, 0),
        F5 => ("F5".into(), "F5".into(), 116, 0),
        F6 => ("F6".into(), "F6".into(), 117, 0),
        F7 => ("F7".into(), "F7".into(), 118, 0),
        F8 => ("F8".into(), "F8".into(), 119, 0),
        F9 => ("F9".into(), "F9".into(), 120, 0),
        F10 => ("F10".into(), "F10".into(), 121, 0),
        F11 => ("F11".into(), "F11".into(), 122, 0),
        F12 => ("F12".into(), "F12".into(), 123, 0),

        // Arrow keys
        ArrowUp => ("ArrowUp".into(), "ArrowUp".into(), 38, 0),
        ArrowDown => ("ArrowDown".into(), "ArrowDown".into(), 40, 0),
        ArrowLeft => ("ArrowLeft".into(), "ArrowLeft".into(), 37, 0),
        ArrowRight => ("ArrowRight".into(), "ArrowRight".into(), 39, 0),

        // Modifiers
        ShiftLeft => ("Shift".into(), "ShiftLeft".into(), 16, 1),
        ShiftRight => ("Shift".into(), "ShiftRight".into(), 16, 2),
        ControlLeft => ("Control".into(), "ControlLeft".into(), 17, 1),
        ControlRight => ("Control".into(), "ControlRight".into(), 17, 2),
        AltLeft => ("Alt".into(), "AltLeft".into(), 18, 1),
        AltRight => ("Alt".into(), "AltRight".into(), 18, 2),
        SuperLeft => ("Meta".into(), "MetaLeft".into(), 91, 1),
        SuperRight => ("Meta".into(), "MetaRight".into(), 92, 2),

        // Special keys
        Escape => ("Escape".into(), "Escape".into(), 27, 0),
        Enter => ("Enter".into(), "Enter".into(), 13, 0),
        Tab => ("Tab".into(), "Tab".into(), 9, 0),
        Space => (" ".into(), "Space".into(), 32, 0),
        Backspace => ("Backspace".into(), "Backspace".into(), 8, 0),
        Delete => ("Delete".into(), "Delete".into(), 46, 0),
        Insert => ("Insert".into(), "Insert".into(), 45, 0),
        Home => ("Home".into(), "Home".into(), 36, 0),
        End => ("End".into(), "End".into(), 35, 0),
        PageUp => ("PageUp".into(), "PageUp".into(), 33, 0),
        PageDown => ("PageDown".into(), "PageDown".into(), 34, 0),
        CapsLock => ("CapsLock".into(), "CapsLock".into(), 20, 0),

        // Punctuation
        Minus => ("-".into(), "Minus".into(), 189, 0),
        Equal => ("=".into(), "Equal".into(), 187, 0),
        BracketLeft => ("[".into(), "BracketLeft".into(), 219, 0),
        BracketRight => ("]".into(), "BracketRight".into(), 221, 0),
        Backslash => ("\\".into(), "Backslash".into(), 220, 0),
        Semicolon => (";".into(), "Semicolon".into(), 186, 0),
        Quote => ("'".into(), "Quote".into(), 222, 0),
        Backquote => ("`".into(), "Backquote".into(), 192, 0),
        Comma => (",".into(), "Comma".into(), 188, 0),
        Period => (".".into(), "Period".into(), 190, 0),
        Slash => ("/".into(), "Slash".into(), 191, 0),

        // Numpad
        Numpad0 => ("0".into(), "Numpad0".into(), 96, 3),
        Numpad1 => ("1".into(), "Numpad1".into(), 97, 3),
        Numpad2 => ("2".into(), "Numpad2".into(), 98, 3),
        Numpad3 => ("3".into(), "Numpad3".into(), 99, 3),
        Numpad4 => ("4".into(), "Numpad4".into(), 100, 3),
        Numpad5 => ("5".into(), "Numpad5".into(), 101, 3),
        Numpad6 => ("6".into(), "Numpad6".into(), 102, 3),
        Numpad7 => ("7".into(), "Numpad7".into(), 103, 3),
        Numpad8 => ("8".into(), "Numpad8".into(), 104, 3),
        Numpad9 => ("9".into(), "Numpad9".into(), 105, 3),
        NumpadAdd => ("+".into(), "NumpadAdd".into(), 107, 3),
        NumpadSubtract => ("-".into(), "NumpadSubtract".into(), 109, 3),
        NumpadMultiply => ("*".into(), "NumpadMultiply".into(), 106, 3),
        NumpadDivide => ("/".into(), "NumpadDivide".into(), 111, 3),
        NumpadDecimal => (".".into(), "NumpadDecimal".into(), 110, 3),
        NumpadEnter => ("Enter".into(), "NumpadEnter".into(), 13, 3),

        // Default
        _ => ("Unidentified".into(), "Unidentified".into(), 0, 0),
    }
}

// ======================= Deno Ops =======================

/// Poll for pending input events
#[op2]
#[serde]
pub fn op_input_poll_events() -> Vec<InputEvent> {
    let queue = get_input_queue();
    let mut guard = queue.lock().unwrap();
    guard.drain_events()
}

/// Get the current window size
#[op2]
#[serde]
pub fn op_input_get_window_size() -> (u32, u32) {
    let queue = get_input_queue();
    let guard = queue.lock().unwrap();
    (guard.window_width, guard.window_height)
}



#[op2(fast)]
pub fn op_input_request_pointer_lock() -> bool {
    let queue = get_input_queue();
    if let Ok(mut q) = queue.lock() {
        q.pointer_lock_requested = true;
        return true;
    };
    false
}

#[op2(fast)]
pub fn op_input_exit_pointer_lock() {
    let queue = get_input_queue();
    if let Ok(mut q) = queue.lock() {
        q.pointer_lock_exit_requested = true;
    };
}

#[op2(fast)]
pub fn op_input_is_pointer_locked() -> bool {
    let queue = get_input_queue();
    if let Ok(q) = queue.lock() {
        return q.pointer_locked;
    }
    false
}

#[op2(fast)]
pub fn op_input_set_cursor_style(#[string] style: &str) {
    let queue = get_input_queue();
    if let Ok(mut q) = queue.lock() {
        q.cursor_style_requested = Some(style.to_string());
    };
}