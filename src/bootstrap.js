// bootstrap.js
//
// Main bootstrap file for the gfx_host extension
// Exports all graphics and input ops to JavaScript

import {
  op_gfx_get_preferred_surface_format,
  op_gfx_surface_configure,
  op_gfx_device_create_shader,
  op_gfx_device_create_buffer_init,
  op_gfx_device_create_buffer,
  op_gfx_queue_write_buffer,
  op_gfx_device_create_texture,
  op_gfx_texture_create_view,
  op_gfx_device_create_sampler,
  op_gfx_device_create_bind_group_layout,
  op_gfx_device_create_pipeline_layout,
  op_gfx_device_create_bind_group,
  op_gfx_device_create_pipeline,
  op_gfx_surface_draw,
  op_gfx_decode_image,
  op_gfx_write_texture_image,
  op_gfx_pipeline_get_bind_group_layout,
  op_gfx_render_to_texture,
  op_gfx_copy_texture_to_texture,
  op_gfx_decode_image_store,
  op_gfx_upload_decoded_image_to_texture,
  op_gfx_decoded_image_drop,
  op_gfx_render_depth_only,
  op_gfx_resource_drop,
  op_gfx_render_xr_frame,

  // Input ops
  op_input_poll_events,
  op_input_get_window_size,
  op_input_request_pointer_lock,
  op_input_exit_pointer_lock,
  op_input_is_pointer_locked,
  op_input_set_cursor_style,

  // Indexed DB
  op_indexeddb_open,
  op_indexeddb_get,
  op_indexeddb_put,
  op_indexeddb_delete,
  op_indexeddb_get_all_keys,
  op_indexeddb_clear,
  op_indexeddb_store_exists,

  // XR
  op_xr_is_supported,
  op_xr_request_session,
  op_xr_wait_frame,
  op_xr_get_viewer_pose,
  op_xr_acquire_swapchain_image,
  op_xr_release_swapchain_image,
  op_xr_end_frame,
  op_xr_end_session,
  op_xr_poll_events,
  op_xr_get_swapchain_texture_view,
  op_xr_get_input_sources,
} from "ext:core/ops";

globalThis.__xr = {
  op_xr_is_supported,
  op_xr_request_session,
  op_xr_wait_frame,
  op_xr_get_viewer_pose,
  op_xr_acquire_swapchain_image,
  op_xr_release_swapchain_image,
  op_xr_end_frame,
  op_xr_end_session,
  op_xr_poll_events,
  op_xr_get_swapchain_texture_view,
  op_xr_get_input_sources,
};

globalThis.__indexeddb = {
  op_indexeddb_open,
  op_indexeddb_get,
  op_indexeddb_put,
  op_indexeddb_delete,
  op_indexeddb_get_all_keys,
  op_indexeddb_clear,
  op_indexeddb_store_exists,
};

// Expose graphics ops to globalThis
globalThis.__gfx = {
  op_gfx_get_preferred_surface_format,
  op_gfx_surface_configure,
  op_gfx_device_create_shader,
  op_gfx_device_create_buffer_init,
  op_gfx_device_create_buffer,
  op_gfx_queue_write_buffer,
  op_gfx_device_create_texture,
  op_gfx_texture_create_view,
  op_gfx_device_create_sampler,
  op_gfx_device_create_bind_group_layout,
  op_gfx_device_create_pipeline_layout,
  op_gfx_device_create_bind_group,
  op_gfx_device_create_pipeline,
  op_gfx_surface_draw,
  op_gfx_decode_image,
  op_gfx_write_texture_image,
  op_gfx_pipeline_get_bind_group_layout,
  op_gfx_render_to_texture,
  op_gfx_copy_texture_to_texture,
  op_gfx_decode_image_store,
  op_gfx_upload_decoded_image_to_texture,
  op_gfx_decoded_image_drop,
  op_gfx_render_depth_only,
  op_gfx_resource_drop,
  op_gfx_render_xr_frame,
};

// Expose input ops to globalThis
globalThis.__input = {
  op_input_poll_events,
  op_input_get_window_size,
  op_input_request_pointer_lock,
  op_input_exit_pointer_lock,
  op_input_is_pointer_locked,
  op_input_set_cursor_style,
};
