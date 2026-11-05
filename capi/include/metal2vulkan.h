// metal2vulkan's C ABI: the Metal API on Vulkan, for programs not written in
// Rust. Built as libmetal2vulkan.so.
//
// Every object is an opaque handle, made by a call named for Metal's
// selector and freed by its _release. Handles are retains: releasing one
// leaves the others to the object. A call that can fail takes `char **error`
// last; on failure it returns NULL, 0 or false, and stores a message there
// (when error is not NULL) that m2v_string_free frees.
//
// SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later

#ifndef METAL2VULKAN_H
#define METAL2VULKAN_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// This header's ABI version, which m2v_abi_version returns.
#define M2V_ABI_VERSION 1

typedef struct m2v_device m2v_device;
typedef struct m2v_buffer m2v_buffer;
typedef struct m2v_library m2v_library;
typedef struct m2v_function m2v_function;
typedef struct m2v_compute_pipeline m2v_compute_pipeline;
typedef struct m2v_command_queue m2v_command_queue;
typedef struct m2v_command_buffer m2v_command_buffer;
typedef struct m2v_compute_encoder m2v_compute_encoder;

/// MTLSize.
typedef struct m2v_size {
    uint64_t width, height, depth;
} m2v_size;

/// Metal's device queries.
typedef struct m2v_device_properties {
    uint64_t registry_id;
    uint64_t max_buffer_length;
    m2v_size max_threads_per_threadgroup;
    uint64_t max_threadgroup_memory_length;
    /// 0 discrete, 1 integrated, 2 virtual, 3 CPU, 4 other.
    uint32_t kind;
    bool has_unified_memory;
    /// MTLArgumentBuffersTier and MTLReadWriteTextureTier.
    uint32_t argument_buffers_support;
    uint32_t read_write_texture_support;
    uint32_t vendor_id;
    uint32_t device_id;
    uint8_t device_uuid[16];
} m2v_device_properties;

/// The library's ABI version.
uint32_t m2v_abi_version(void);
void m2v_string_free(char *string);

// MARK: Devices

/// MTLCopyAllDevices: every device, preferred first. Stores at most
/// `capacity` handles and returns how many there are, or -1 on failure.
int64_t m2v_copy_all_devices(m2v_device **devices, size_t capacity, char **error);
m2v_device *m2v_device_retain(m2v_device *device);
void m2v_device_release(m2v_device *device);
/// Whether two handles are the same device.
bool m2v_device_same(const m2v_device *a, const m2v_device *b);
/// The name, valid while the device is.
const char *m2v_device_name(const m2v_device *device);
void m2v_device_properties_get(const m2v_device *device, m2v_device_properties *properties);
bool m2v_device_supports_family(const m2v_device *device, int64_t family);
/// 1 or 0; -1 when the runtime cannot say.
int m2v_device_supports_texture_sample_count(const m2v_device *device, uint64_t count);

// MARK: Buffers

/// newBufferWithLength:options: or, with bytes, newBufferWithBytes:length:options:.
m2v_buffer *m2v_device_new_buffer(const m2v_device *device, const void *bytes, uint64_t length,
                                  uint64_t options, char **error);
void m2v_buffer_release(m2v_buffer *buffer);
uint64_t m2v_buffer_length(const m2v_buffer *buffer);
/// The bytes the host reaches, or NULL for private storage.
void *m2v_buffer_contents(const m2v_buffer *buffer);
bool m2v_buffer_did_modify_range(const m2v_buffer *buffer, uint64_t location, uint64_t length, char **error);

// MARK: Libraries and functions

/// newLibraryWithData:error:, from a metallib.
m2v_library *m2v_device_new_library_with_data(const m2v_device *device, const void *data, size_t length,
                                              char **error);
void m2v_library_release(m2v_library *library);
/// functionNames: returns how many, and stores a malloc'd array of
/// malloc'd strings in *names, which m2v_strings_free frees.
size_t m2v_library_function_names(const m2v_library *library, char ***names);
void m2v_strings_free(char **strings, size_t count);
m2v_function *m2v_library_new_function_with_name(const m2v_library *library, const char *name, char **error);
void m2v_function_release(m2v_function *function);
/// The name, valid while the function is.
const char *m2v_function_name(const m2v_function *function);
/// MTLFunctionType, or 0 when the runtime cannot say.
uint32_t m2v_function_type(const m2v_function *function);

// MARK: Compute pipelines

m2v_compute_pipeline *m2v_device_new_compute_pipeline_state_with_function(const m2v_device *device,
                                                                         const m2v_function *function,
                                                                         char **error);
void m2v_compute_pipeline_release(m2v_compute_pipeline *pipeline);
uint32_t m2v_compute_pipeline_max_total_threads_per_threadgroup(const m2v_compute_pipeline *pipeline);
uint32_t m2v_compute_pipeline_thread_execution_width(const m2v_compute_pipeline *pipeline);
uint32_t m2v_compute_pipeline_static_threadgroup_memory_length(const m2v_compute_pipeline *pipeline);

// MARK: Queues and command buffers

m2v_command_queue *m2v_device_new_command_queue(const m2v_device *device, char **error);
void m2v_command_queue_release(m2v_command_queue *queue);
m2v_command_buffer *m2v_command_queue_command_buffer(const m2v_command_queue *queue, char **error);
void m2v_command_buffer_release(m2v_command_buffer *buffer);
bool m2v_command_buffer_commit(const m2v_command_buffer *buffer, char **error);
bool m2v_command_buffer_wait_until_scheduled(const m2v_command_buffer *buffer, char **error);
bool m2v_command_buffer_wait_until_completed(const m2v_command_buffer *buffer, char **error);
/// MTLCommandBufferStatus.
uint32_t m2v_command_buffer_status(const m2v_command_buffer *buffer);
/// Why it failed, malloc'd, or NULL.
char *m2v_command_buffer_error(const m2v_command_buffer *buffer);
/// Called once, on a thread of the runtime's, with `context`.
typedef void (*m2v_handler)(void *context);
bool m2v_command_buffer_add_scheduled_handler(const m2v_command_buffer *buffer, m2v_handler handler,
                                              void *context, char **error);
bool m2v_command_buffer_add_completed_handler(const m2v_command_buffer *buffer, m2v_handler handler,
                                              void *context, char **error);

// MARK: Compute encoders

/// An encoder belongs to one thread at a time. m2v_compute_encoder_end
/// frees it; m2v_compute_encoder_release frees one never ended.
m2v_compute_encoder *m2v_command_buffer_compute_command_encoder(const m2v_command_buffer *buffer, char **error);
void m2v_compute_encoder_release(m2v_compute_encoder *encoder);
bool m2v_compute_encoder_set_compute_pipeline_state(m2v_compute_encoder *encoder,
                                                    const m2v_compute_pipeline *pipeline, char **error);
/// A NULL buffer unbinds the slot.
bool m2v_compute_encoder_set_buffer(m2v_compute_encoder *encoder, const m2v_buffer *buffer, uint64_t offset,
                                    uint32_t index, char **error);
bool m2v_compute_encoder_set_buffer_offset(m2v_compute_encoder *encoder, uint64_t offset, uint32_t index,
                                           char **error);
bool m2v_compute_encoder_set_bytes(m2v_compute_encoder *encoder, const void *bytes, size_t length, uint32_t index,
                                   char **error);
bool m2v_compute_encoder_set_threadgroup_memory_length(m2v_compute_encoder *encoder, uint32_t length,
                                                       uint32_t index, char **error);
bool m2v_compute_encoder_dispatch_threads(m2v_compute_encoder *encoder, m2v_size threads, m2v_size per_group,
                                          char **error);
bool m2v_compute_encoder_dispatch_threadgroups(m2v_compute_encoder *encoder, m2v_size groups, m2v_size per_group,
                                               char **error);
bool m2v_compute_encoder_end(m2v_compute_encoder *encoder, char **error);

#ifdef __cplusplus
}
#endif

#endif
