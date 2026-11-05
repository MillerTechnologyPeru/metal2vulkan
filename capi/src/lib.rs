//! metal2vulkan's C ABI (`include/metal2vulkan.h`): each Metal object of the runtime behind an
//! opaque handle, a boxed clone of it (a retain), freed by its `_release`.
//!
//! SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later

#![allow(clippy::missing_safety_doc)]

use metal2vulkan_runtime::{
    Buffer, CommandBuffer, CommandBufferStatus, CommandQueue, ComputeCommandEncoder,
    ComputePipelineState, Device, DeviceKind, Function, Library, PipelineOptions, ResourceOptions,
    Size,
};
use std::ffi::{c_char, c_void, CStr, CString};
use std::fmt::Display;
use std::ptr;

pub const ABI_VERSION: u32 = 1;

pub struct M2vDevice {
    device: Device,
    name: CString,
}

pub struct M2vBuffer(Buffer);
pub struct M2vLibrary(Library);

pub struct M2vFunction {
    function: Function,
    name: CString,
}

pub struct M2vComputePipeline(ComputePipelineState);
pub struct M2vCommandQueue(CommandQueue);
pub struct M2vCommandBuffer(CommandBuffer);
pub struct M2vComputeEncoder(Option<ComputeCommandEncoder>);

#[repr(C)]
#[derive(Clone, Copy)]
pub struct M2vSize {
    width: u64,
    height: u64,
    depth: u64,
}

impl From<M2vSize> for Size {
    fn from(size: M2vSize) -> Size {
        Size::new(size.width, size.height, size.depth)
    }
}

#[repr(C)]
pub struct M2vDeviceProperties {
    registry_id: u64,
    max_buffer_length: u64,
    max_threads_per_threadgroup: M2vSize,
    max_threadgroup_memory_length: u64,
    kind: u32,
    has_unified_memory: bool,
    argument_buffers_support: u32,
    read_write_texture_support: u32,
    vendor_id: u32,
    device_id: u32,
    device_uuid: [u8; 16],
}

/// A message for `*error`, which `m2v_string_free` frees.
fn c_string(text: impl Display) -> *mut c_char {
    let text = text.to_string().replace('\0', " ");
    CString::new(text).map(CString::into_raw).unwrap_or(ptr::null_mut())
}

unsafe fn fail<T>(error: *mut *mut c_char, message: impl Display, value: T) -> T {
    if !error.is_null() {
        *error = c_string(message);
    }
    value
}

/// The handle's value, or a failure for a NULL handle.
macro_rules! handle {
    ($pointer:expr, $error:expr, $failed:expr) => {
        match $pointer.as_ref() {
            Some(value) => value,
            None => return fail($error, concat!(stringify!($pointer), " is NULL"), $failed),
        }
    };
}

/// A handle's result, or its error.
macro_rules! check {
    ($result:expr, $error:expr, $failed:expr) => {
        match $result {
            Ok(value) => value,
            Err(why) => return fail($error, why, $failed),
        }
    };
}

fn boxed<T>(value: T) -> *mut T {
    Box::into_raw(Box::new(value))
}

unsafe fn release<T>(pointer: *mut T) {
    if !pointer.is_null() {
        drop(Box::from_raw(pointer));
    }
}

fn device_handle(device: Device) -> *mut M2vDevice {
    let name = CString::new(device.name().replace('\0', " ")).unwrap_or_default();
    boxed(M2vDevice { device, name })
}

#[no_mangle]
pub extern "C" fn m2v_abi_version() -> u32 {
    ABI_VERSION
}

#[no_mangle]
pub unsafe extern "C" fn m2v_string_free(string: *mut c_char) {
    if !string.is_null() {
        drop(CString::from_raw(string));
    }
}

// MARK: Devices

#[no_mangle]
pub unsafe extern "C" fn m2v_copy_all_devices(
    devices: *mut *mut M2vDevice,
    capacity: usize,
    error: *mut *mut c_char,
) -> i64 {
    let all = check!(Device::all(), error, -1);
    let count = all.len();
    for (i, device) in all.into_iter().enumerate().take(if devices.is_null() { 0 } else { capacity }) {
        *devices.add(i) = device_handle(device);
    }
    count as i64
}

#[no_mangle]
pub unsafe extern "C" fn m2v_device_retain(device: *mut M2vDevice) -> *mut M2vDevice {
    match device.as_ref() {
        Some(device) => device_handle(device.device.clone()),
        None => ptr::null_mut(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn m2v_device_release(device: *mut M2vDevice) {
    release(device)
}

#[no_mangle]
pub unsafe extern "C" fn m2v_device_same(a: *const M2vDevice, b: *const M2vDevice) -> bool {
    match (a.as_ref(), b.as_ref()) {
        // One device per physical device and options while a handle lives.
        (Some(a), Some(b)) => a.device.registry_id() == b.device.registry_id(),
        _ => false,
    }
}

#[no_mangle]
pub unsafe extern "C" fn m2v_device_name(device: *const M2vDevice) -> *const c_char {
    device.as_ref().map_or(ptr::null(), |device| device.name.as_ptr())
}

#[no_mangle]
pub unsafe extern "C" fn m2v_device_properties_get(
    device: *const M2vDevice,
    properties: *mut M2vDeviceProperties,
) {
    let (Some(device), Some(out)) = (device.as_ref(), properties.as_mut()) else { return };
    let p = device.device.properties();
    let threads = p.max_threads_per_threadgroup;
    *out = M2vDeviceProperties {
        registry_id: device.device.registry_id(),
        max_buffer_length: p.max_buffer_length,
        max_threads_per_threadgroup: M2vSize {
            width: threads.width,
            height: threads.height,
            depth: threads.depth,
        },
        max_threadgroup_memory_length: p.max_threadgroup_memory_length,
        kind: match p.kind {
            DeviceKind::Discrete => 0,
            DeviceKind::Integrated => 1,
            DeviceKind::Virtual => 2,
            DeviceKind::Cpu => 3,
            _ => 4,
        },
        has_unified_memory: p.has_unified_memory,
        argument_buffers_support: p.argument_buffers_support.as_raw(),
        read_write_texture_support: p.read_write_texture_support.as_raw(),
        vendor_id: p.vendor_id,
        device_id: p.device_id,
        device_uuid: p.device_uuid,
    };
}

#[no_mangle]
pub unsafe extern "C" fn m2v_device_supports_family(device: *const M2vDevice, family: i64) -> bool {
    device.as_ref().is_some_and(|device| device.device.supports_family(family))
}

#[no_mangle]
pub unsafe extern "C" fn m2v_device_supports_texture_sample_count(
    device: *const M2vDevice,
    count: u64,
) -> i32 {
    match device.as_ref().map(|device| device.device.properties().supports_texture_sample_count(count)) {
        Some(Ok(true)) => 1,
        Some(Ok(false)) => 0,
        _ => -1,
    }
}

// MARK: Buffers

#[no_mangle]
pub unsafe extern "C" fn m2v_device_new_buffer(
    device: *const M2vDevice,
    bytes: *const c_void,
    length: u64,
    options: u64,
    error: *mut *mut c_char,
) -> *mut M2vBuffer {
    let device = handle!(device, error, ptr::null_mut());
    let options = check!(ResourceOptions::from_raw(options), error, ptr::null_mut());
    let buffer = if bytes.is_null() {
        device.device.new_buffer(length, options)
    } else {
        let bytes = std::slice::from_raw_parts(bytes.cast::<u8>(), length as usize);
        device.device.new_buffer_with_bytes(bytes, options)
    };
    boxed(M2vBuffer(check!(buffer, error, ptr::null_mut())))
}

#[no_mangle]
pub unsafe extern "C" fn m2v_buffer_release(buffer: *mut M2vBuffer) {
    release(buffer)
}

#[no_mangle]
pub unsafe extern "C" fn m2v_buffer_length(buffer: *const M2vBuffer) -> u64 {
    buffer.as_ref().map_or(0, |buffer| buffer.0.length())
}

#[no_mangle]
pub unsafe extern "C" fn m2v_buffer_contents(buffer: *const M2vBuffer) -> *mut c_void {
    buffer
        .as_ref()
        .and_then(|buffer| buffer.0.contents())
        .map_or(ptr::null_mut(), |contents| contents.as_ptr().cast())
}

#[no_mangle]
pub unsafe extern "C" fn m2v_buffer_did_modify_range(
    buffer: *const M2vBuffer,
    location: u64,
    length: u64,
    error: *mut *mut c_char,
) -> bool {
    let buffer = handle!(buffer, error, false);
    check!(buffer.0.did_modify_range(location..location + length), error, false);
    true
}

// MARK: Libraries and functions

#[no_mangle]
pub unsafe extern "C" fn m2v_device_new_library_with_data(
    device: *const M2vDevice,
    data: *const c_void,
    length: usize,
    error: *mut *mut c_char,
) -> *mut M2vLibrary {
    let device = handle!(device, error, ptr::null_mut());
    if data.is_null() {
        return fail(error, "data is NULL", ptr::null_mut());
    }
    let data = std::slice::from_raw_parts(data.cast::<u8>(), length);
    boxed(M2vLibrary(check!(device.device.new_library_with_data(data), error, ptr::null_mut())))
}

#[no_mangle]
pub unsafe extern "C" fn m2v_library_release(library: *mut M2vLibrary) {
    release(library)
}

#[no_mangle]
pub unsafe extern "C" fn m2v_library_function_names(
    library: *const M2vLibrary,
    names: *mut *mut *mut c_char,
) -> usize {
    let Some(library) = library.as_ref() else { return 0 };
    let list: Vec<*mut c_char> = library.0.function_names().into_iter().map(c_string).collect();
    let count = list.len();
    if names.is_null() {
        for name in list {
            m2v_string_free(name);
        }
    } else {
        *names = Box::into_raw(list.into_boxed_slice()).cast();
    }
    count
}

#[no_mangle]
pub unsafe extern "C" fn m2v_strings_free(strings: *mut *mut c_char, count: usize) {
    if strings.is_null() {
        return;
    }
    let list = Box::from_raw(ptr::slice_from_raw_parts_mut(strings, count));
    for &name in list.iter() {
        m2v_string_free(name);
    }
}

#[no_mangle]
pub unsafe extern "C" fn m2v_library_new_function_with_name(
    library: *const M2vLibrary,
    name: *const c_char,
    error: *mut *mut c_char,
) -> *mut M2vFunction {
    let library = handle!(library, error, ptr::null_mut());
    if name.is_null() {
        return fail(error, "name is NULL", ptr::null_mut());
    }
    let name = CStr::from_ptr(name);
    let text = check!(name.to_str(), error, ptr::null_mut());
    let function = check!(library.0.new_function_with_name(text), error, ptr::null_mut());
    boxed(M2vFunction { function, name: name.to_owned() })
}

#[no_mangle]
pub unsafe extern "C" fn m2v_function_release(function: *mut M2vFunction) {
    release(function)
}

#[no_mangle]
pub unsafe extern "C" fn m2v_function_name(function: *const M2vFunction) -> *const c_char {
    function.as_ref().map_or(ptr::null(), |function| function.name.as_ptr())
}

#[no_mangle]
pub unsafe extern "C" fn m2v_function_type(function: *const M2vFunction) -> u32 {
    function
        .as_ref()
        .and_then(|function| function.function.function_type().ok())
        .map_or(0, |kind| kind.as_raw())
}

// MARK: Compute pipelines

#[no_mangle]
pub unsafe extern "C" fn m2v_device_new_compute_pipeline_state_with_function(
    device: *const M2vDevice,
    function: *const M2vFunction,
    error: *mut *mut c_char,
) -> *mut M2vComputePipeline {
    let device = handle!(device, error, ptr::null_mut());
    let function = handle!(function, error, ptr::null_mut());
    let state = device
        .device
        .new_compute_pipeline_state_with_function(&function.function, &PipelineOptions::default());
    boxed(M2vComputePipeline(check!(state, error, ptr::null_mut())))
}

#[no_mangle]
pub unsafe extern "C" fn m2v_compute_pipeline_release(pipeline: *mut M2vComputePipeline) {
    release(pipeline)
}

#[no_mangle]
pub unsafe extern "C" fn m2v_compute_pipeline_max_total_threads_per_threadgroup(
    pipeline: *const M2vComputePipeline,
) -> u32 {
    pipeline.as_ref().map_or(0, |p| p.0.max_total_threads_per_threadgroup())
}

#[no_mangle]
pub unsafe extern "C" fn m2v_compute_pipeline_thread_execution_width(
    pipeline: *const M2vComputePipeline,
) -> u32 {
    pipeline.as_ref().map_or(0, |p| p.0.thread_execution_width())
}

#[no_mangle]
pub unsafe extern "C" fn m2v_compute_pipeline_static_threadgroup_memory_length(
    pipeline: *const M2vComputePipeline,
) -> u32 {
    pipeline.as_ref().map_or(0, |p| p.0.static_threadgroup_memory_length())
}

// MARK: Queues and command buffers

#[no_mangle]
pub unsafe extern "C" fn m2v_device_new_command_queue(
    device: *const M2vDevice,
    error: *mut *mut c_char,
) -> *mut M2vCommandQueue {
    let device = handle!(device, error, ptr::null_mut());
    boxed(M2vCommandQueue(check!(device.device.new_command_queue(), error, ptr::null_mut())))
}

#[no_mangle]
pub unsafe extern "C" fn m2v_command_queue_release(queue: *mut M2vCommandQueue) {
    release(queue)
}

#[no_mangle]
pub unsafe extern "C" fn m2v_command_queue_command_buffer(
    queue: *const M2vCommandQueue,
    error: *mut *mut c_char,
) -> *mut M2vCommandBuffer {
    let queue = handle!(queue, error, ptr::null_mut());
    boxed(M2vCommandBuffer(check!(queue.0.command_buffer(), error, ptr::null_mut())))
}

#[no_mangle]
pub unsafe extern "C" fn m2v_command_buffer_release(buffer: *mut M2vCommandBuffer) {
    release(buffer)
}

#[no_mangle]
pub unsafe extern "C" fn m2v_command_buffer_commit(
    buffer: *const M2vCommandBuffer,
    error: *mut *mut c_char,
) -> bool {
    let buffer = handle!(buffer, error, false);
    check!(buffer.0.commit(), error, false);
    true
}

#[no_mangle]
pub unsafe extern "C" fn m2v_command_buffer_wait_until_scheduled(
    buffer: *const M2vCommandBuffer,
    error: *mut *mut c_char,
) -> bool {
    let buffer = handle!(buffer, error, false);
    check!(buffer.0.wait_until_scheduled(), error, false);
    true
}

#[no_mangle]
pub unsafe extern "C" fn m2v_command_buffer_wait_until_completed(
    buffer: *const M2vCommandBuffer,
    error: *mut *mut c_char,
) -> bool {
    let buffer = handle!(buffer, error, false);
    check!(buffer.0.wait_until_completed(), error, false);
    true
}

#[no_mangle]
pub unsafe extern "C" fn m2v_command_buffer_status(buffer: *const M2vCommandBuffer) -> u32 {
    match buffer.as_ref().map(|buffer| buffer.0.status()) {
        Some(CommandBufferStatus::NotEnqueued) | None => 0,
        Some(CommandBufferStatus::Enqueued) => 1,
        Some(CommandBufferStatus::Committed) => 2,
        Some(CommandBufferStatus::Scheduled) => 3,
        Some(CommandBufferStatus::Completed) => 4,
        Some(CommandBufferStatus::Error) => 5,
        #[allow(unreachable_patterns)]
        Some(_) => 5,
    }
}

#[no_mangle]
pub unsafe extern "C" fn m2v_command_buffer_error(buffer: *const M2vCommandBuffer) -> *mut c_char {
    buffer
        .as_ref()
        .and_then(|buffer| buffer.0.error())
        .map_or(ptr::null_mut(), |why| c_string(format!("{why:?}")))
}

/// A C callback and its context, called once on a thread of the runtime's.
struct Callback(extern "C" fn(*mut c_void), *mut c_void);
// SAFETY: the C caller hands the context over to the runtime's thread (the header's contract).
unsafe impl Send for Callback {}

impl Callback {
    fn handler(self) -> metal2vulkan_runtime::Handler {
        Box::new(move |_| {
            let callback = self;
            (callback.0)(callback.1)
        })
    }
}

#[no_mangle]
pub unsafe extern "C" fn m2v_command_buffer_add_scheduled_handler(
    buffer: *const M2vCommandBuffer,
    handler: Option<extern "C" fn(*mut c_void)>,
    context: *mut c_void,
    error: *mut *mut c_char,
) -> bool {
    let buffer = handle!(buffer, error, false);
    let Some(handler) = handler else { return fail(error, "handler is NULL", false) };
    check!(buffer.0.add_scheduled_handler(Callback(handler, context).handler()), error, false);
    true
}

#[no_mangle]
pub unsafe extern "C" fn m2v_command_buffer_add_completed_handler(
    buffer: *const M2vCommandBuffer,
    handler: Option<extern "C" fn(*mut c_void)>,
    context: *mut c_void,
    error: *mut *mut c_char,
) -> bool {
    let buffer = handle!(buffer, error, false);
    let Some(handler) = handler else { return fail(error, "handler is NULL", false) };
    check!(buffer.0.add_completed_handler(Callback(handler, context).handler()), error, false);
    true
}

// MARK: Compute encoders

#[no_mangle]
pub unsafe extern "C" fn m2v_command_buffer_compute_command_encoder(
    buffer: *const M2vCommandBuffer,
    error: *mut *mut c_char,
) -> *mut M2vComputeEncoder {
    let buffer = handle!(buffer, error, ptr::null_mut());
    let encoder = check!(buffer.0.compute_command_encoder(), error, ptr::null_mut());
    boxed(M2vComputeEncoder(Some(encoder)))
}

#[no_mangle]
pub unsafe extern "C" fn m2v_compute_encoder_release(encoder: *mut M2vComputeEncoder) {
    release(encoder)
}

/// The encoder, or a failure for a NULL or ended one.
macro_rules! encoder {
    ($encoder:expr, $error:expr) => {
        match $encoder.as_mut().and_then(|encoder| encoder.0.as_mut()) {
            Some(encoder) => encoder,
            None => return fail($error, "the encoder is NULL or ended", false),
        }
    };
}

#[no_mangle]
pub unsafe extern "C" fn m2v_compute_encoder_set_compute_pipeline_state(
    encoder: *mut M2vComputeEncoder,
    pipeline: *const M2vComputePipeline,
    error: *mut *mut c_char,
) -> bool {
    let encoder = encoder!(encoder, error);
    let pipeline = handle!(pipeline, error, false);
    check!(encoder.set_compute_pipeline_state(&pipeline.0), error, false);
    true
}

#[no_mangle]
pub unsafe extern "C" fn m2v_compute_encoder_set_buffer(
    encoder: *mut M2vComputeEncoder,
    buffer: *const M2vBuffer,
    offset: u64,
    index: u32,
    error: *mut *mut c_char,
) -> bool {
    let encoder = encoder!(encoder, error);
    check!(encoder.set_buffer(buffer.as_ref().map(|buffer| &buffer.0), offset, index), error, false);
    true
}

#[no_mangle]
pub unsafe extern "C" fn m2v_compute_encoder_set_buffer_offset(
    encoder: *mut M2vComputeEncoder,
    offset: u64,
    index: u32,
    error: *mut *mut c_char,
) -> bool {
    let encoder = encoder!(encoder, error);
    check!(encoder.set_buffer_offset(offset, index), error, false);
    true
}

#[no_mangle]
pub unsafe extern "C" fn m2v_compute_encoder_set_bytes(
    encoder: *mut M2vComputeEncoder,
    bytes: *const c_void,
    length: usize,
    index: u32,
    error: *mut *mut c_char,
) -> bool {
    let encoder = encoder!(encoder, error);
    if bytes.is_null() {
        return fail(error, "bytes is NULL", false);
    }
    let bytes = std::slice::from_raw_parts(bytes.cast::<u8>(), length);
    check!(encoder.set_bytes(bytes, index), error, false);
    true
}

#[no_mangle]
pub unsafe extern "C" fn m2v_compute_encoder_set_threadgroup_memory_length(
    encoder: *mut M2vComputeEncoder,
    length: u32,
    index: u32,
    error: *mut *mut c_char,
) -> bool {
    let encoder = encoder!(encoder, error);
    check!(encoder.set_threadgroup_memory_length(length, index), error, false);
    true
}

#[no_mangle]
pub unsafe extern "C" fn m2v_compute_encoder_dispatch_threads(
    encoder: *mut M2vComputeEncoder,
    threads: M2vSize,
    per_group: M2vSize,
    error: *mut *mut c_char,
) -> bool {
    let encoder = encoder!(encoder, error);
    check!(encoder.dispatch_threads(threads.into(), per_group.into()), error, false);
    true
}

#[no_mangle]
pub unsafe extern "C" fn m2v_compute_encoder_dispatch_threadgroups(
    encoder: *mut M2vComputeEncoder,
    groups: M2vSize,
    per_group: M2vSize,
    error: *mut *mut c_char,
) -> bool {
    let encoder = encoder!(encoder, error);
    check!(encoder.dispatch_threadgroups(groups.into(), per_group.into()), error, false);
    true
}

#[no_mangle]
pub unsafe extern "C" fn m2v_compute_encoder_end(
    encoder: *mut M2vComputeEncoder,
    error: *mut *mut c_char,
) -> bool {
    if encoder.is_null() {
        return fail(error, "the encoder is NULL", false);
    }
    let encoder = Box::from_raw(encoder);
    let Some(encoder) = encoder.0 else { return fail(error, "the encoder already ended", false) };
    check!(encoder.end_encoding(), error, false);
    true
}
