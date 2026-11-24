//! Argument buffers in the C ABI: argument encoders (`MTLArgumentEncoder`), the addresses and
//! resource IDs a program writes into an argument buffer itself, and the `useResource` calls that
//! make what a function reaches through one resident.
//!
//! SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later

use super::render::{M2vRenderEncoder, M2vSamplerState, M2vTexture};
use super::*;
use metal2vulkan::bind::ArgumentDescriptor;
use metal2vulkan::metal::{BindingAccess, RenderStages, ResourceUsage, TextureType};
use metal2vulkan_runtime::{ArgumentEncoder, ResourceRef};

pub struct M2vArgumentEncoder(ArgumentEncoder);

/// A Metal enum or option set from its raw value, or why not.
fn raw<T: TryFrom<u32>>(value: u32, what: &str) -> Result<T, String> {
    T::try_from(value).map_err(|_| format!("{value} is not a valid {what}"))
}

/// `MTLArgumentDescriptor`, enums as Metal's raw values.
#[repr(C)]
pub struct M2vArgumentDescriptor {
    pub data_type: u32,
    pub index: u32,
    pub array_length: u32,
    pub access: u32,
    pub texture_type: u32,
    pub constant_block_alignment: u32,
}

/// `newArgumentEncoderWithArguments:`.
#[no_mangle]
pub unsafe extern "C" fn m2v_device_new_argument_encoder_with_arguments(
    device: *const M2vDevice,
    arguments: *const M2vArgumentDescriptor,
    count: usize,
    error: *mut *mut c_char,
) -> *mut M2vArgumentEncoder {
    let device = handle!(device, error, ptr::null_mut());
    if arguments.is_null() && count > 0 {
        return fail(error, "arguments is NULL", ptr::null_mut());
    }
    let entries = if count == 0 { &[][..] } else { std::slice::from_raw_parts(arguments, count) };
    let mut descriptors = Vec::with_capacity(entries.len());
    for entry in entries {
        let data_type = check!(DataType::try_from(entry.data_type), error, ptr::null_mut());
        let mut descriptor = ArgumentDescriptor::new(data_type, entry.index);
        descriptor.array_length = entry.array_length;
        descriptor.access =
            check!(raw::<BindingAccess>(entry.access, "MTLBindingAccess"), error, ptr::null_mut());
        descriptor.texture_type =
            check!(raw::<TextureType>(entry.texture_type, "MTLTextureType"), error, ptr::null_mut());
        descriptor.constant_block_alignment = entry.constant_block_alignment;
        descriptors.push(descriptor);
    }
    let encoder = check!(
        device.device.new_argument_encoder_with_arguments(&descriptors),
        error,
        ptr::null_mut()
    );
    boxed(M2vArgumentEncoder(encoder))
}

/// `newArgumentEncoderWithBufferIndex:`.
#[no_mangle]
pub unsafe extern "C" fn m2v_function_new_argument_encoder_with_buffer_index(
    function: *const M2vFunction,
    index: u32,
    error: *mut *mut c_char,
) -> *mut M2vArgumentEncoder {
    let function = handle!(function, error, ptr::null_mut());
    let encoder = check!(
        function.function.new_argument_encoder_with_buffer_index(index),
        error,
        ptr::null_mut()
    );
    boxed(M2vArgumentEncoder(encoder))
}

/// `newArgumentEncoderForBufferAtIndex:`.
#[no_mangle]
pub unsafe extern "C" fn m2v_argument_encoder_new_for_buffer_at_index(
    encoder: *const M2vArgumentEncoder,
    index: u32,
    error: *mut *mut c_char,
) -> *mut M2vArgumentEncoder {
    let encoder = handle!(encoder, error, ptr::null_mut());
    let nested =
        check!(encoder.0.new_argument_encoder_for_buffer_at_index(index), error, ptr::null_mut());
    boxed(M2vArgumentEncoder(nested))
}

#[no_mangle]
pub unsafe extern "C" fn m2v_argument_encoder_release(encoder: *mut M2vArgumentEncoder) {
    release(encoder)
}

/// `encodedLength`; 0 for a NULL encoder.
#[no_mangle]
pub unsafe extern "C" fn m2v_argument_encoder_encoded_length(encoder: *const M2vArgumentEncoder) -> u64 {
    encoder.as_ref().map_or(0, |encoder| encoder.0.encoded_length())
}

/// `alignment`; 0 for a NULL encoder.
#[no_mangle]
pub unsafe extern "C" fn m2v_argument_encoder_alignment(encoder: *const M2vArgumentEncoder) -> u64 {
    encoder.as_ref().map_or(0, |encoder| encoder.0.alignment())
}

/// `setArgumentBuffer:offset:`; a NULL buffer sets none.
#[no_mangle]
pub unsafe extern "C" fn m2v_argument_encoder_set_argument_buffer(
    encoder: *mut M2vArgumentEncoder,
    buffer: *const M2vBuffer,
    offset: u64,
    error: *mut *mut c_char,
) -> bool {
    let Some(encoder) = encoder.as_mut() else { return fail(error, "the encoder is NULL", false) };
    check!(encoder.0.set_argument_buffer(buffer.as_ref().map(|b| &b.0), offset), error, false);
    true
}

/// `setArgumentBuffer:startOffset:arrayElement:`.
#[no_mangle]
pub unsafe extern "C" fn m2v_argument_encoder_set_argument_buffer_array_element(
    encoder: *mut M2vArgumentEncoder,
    buffer: *const M2vBuffer,
    start_offset: u64,
    array_element: u64,
    error: *mut *mut c_char,
) -> bool {
    let Some(encoder) = encoder.as_mut() else { return fail(error, "the encoder is NULL", false) };
    check!(
        encoder.0.set_argument_buffer_start_offset_array_element(
            buffer.as_ref().map(|b| &b.0),
            start_offset,
            array_element
        ),
        error,
        false
    );
    true
}

/// `setBuffer:offset:atIndex:`; a NULL buffer writes 0.
#[no_mangle]
pub unsafe extern "C" fn m2v_argument_encoder_set_buffer(
    encoder: *const M2vArgumentEncoder,
    buffer: *const M2vBuffer,
    offset: u64,
    index: u32,
    error: *mut *mut c_char,
) -> bool {
    let encoder = handle!(encoder, error, false);
    check!(encoder.0.set_buffer(buffer.as_ref().map(|b| &b.0), offset, index), error, false);
    true
}

/// `setTexture:atIndex:`; a NULL texture writes 0.
#[no_mangle]
pub unsafe extern "C" fn m2v_argument_encoder_set_texture(
    encoder: *const M2vArgumentEncoder,
    texture: *const M2vTexture,
    index: u32,
    error: *mut *mut c_char,
) -> bool {
    let encoder = handle!(encoder, error, false);
    check!(encoder.0.set_texture(texture.as_ref().map(|t| &t.0), index), error, false);
    true
}

/// `setSamplerState:atIndex:`; a NULL sampler state writes 0.
#[no_mangle]
pub unsafe extern "C" fn m2v_argument_encoder_set_sampler_state(
    encoder: *const M2vArgumentEncoder,
    sampler: *const M2vSamplerState,
    index: u32,
    error: *mut *mut c_char,
) -> bool {
    let encoder = handle!(encoder, error, false);
    check!(encoder.0.set_sampler_state(sampler.as_ref().map(|s| &s.0), index), error, false);
    true
}

/// `constantDataAtIndex:`: where the program writes the constant, in the argument buffer.
#[no_mangle]
pub unsafe extern "C" fn m2v_argument_encoder_constant_data(
    encoder: *const M2vArgumentEncoder,
    index: u32,
    error: *mut *mut c_char,
) -> *mut c_void {
    let encoder = handle!(encoder, error, ptr::null_mut());
    check!(encoder.0.constant_data(index), error, ptr::null_mut()).as_ptr().cast()
}

// MARK: Addresses and resource IDs

/// `gpuAddress`: what a program writes for the buffer in an argument buffer; 0 for NULL.
#[no_mangle]
pub unsafe extern "C" fn m2v_buffer_gpu_address(buffer: *const M2vBuffer) -> u64 {
    buffer.as_ref().map_or(0, |buffer| buffer.0.gpu_address().0)
}

/// `gpuResourceID`; 0 for NULL.
#[no_mangle]
pub unsafe extern "C" fn m2v_texture_gpu_resource_id(texture: *const M2vTexture) -> u64 {
    texture.as_ref().map_or(0, |texture| texture.0.gpu_resource_id().0)
}

/// `gpuResourceID`; 0 for NULL.
#[no_mangle]
pub unsafe extern "C" fn m2v_sampler_state_gpu_resource_id(sampler: *const M2vSamplerState) -> u64 {
    sampler.as_ref().map_or(0, |sampler| sampler.0.gpu_resource_id().0)
}

// MARK: Residency

/// The one resource of a `useResource` call: a buffer or a texture.
unsafe fn resource<'a>(
    buffer: *const M2vBuffer,
    texture: *const M2vTexture,
) -> Result<ResourceRef<'a>, &'static str> {
    match (buffer.as_ref(), texture.as_ref()) {
        (Some(buffer), None) => Ok(ResourceRef::Buffer(&buffer.0)),
        (None, Some(texture)) => Ok(ResourceRef::Texture(&texture.0)),
        _ => Err("exactly one of buffer and texture must be given"),
    }
}

/// `useResource:usage:stages:` of a render encoder, for a buffer or a texture (the other NULL).
#[no_mangle]
pub unsafe extern "C" fn m2v_render_encoder_use_resource(
    encoder: *mut M2vRenderEncoder,
    buffer: *const M2vBuffer,
    texture: *const M2vTexture,
    usage: u32,
    stages: u32,
    error: *mut *mut c_char,
) -> bool {
    let Some(encoder) = encoder.as_mut().and_then(|e| e.0.as_mut()) else {
        return fail(error, "the encoder is NULL or ended", false);
    };
    let resource = check!(resource(buffer, texture), error, false);
    let usage = check!(raw::<ResourceUsage>(usage, "MTLResourceUsage"), error, false);
    let stages = check!(raw::<RenderStages>(stages, "MTLRenderStages"), error, false);
    check!(encoder.use_resource(resource, usage, stages), error, false);
    true
}

/// `useResource:usage:` of a compute encoder, for a buffer or a texture (the other NULL).
#[no_mangle]
pub unsafe extern "C" fn m2v_compute_encoder_use_resource(
    encoder: *mut M2vComputeEncoder,
    buffer: *const M2vBuffer,
    texture: *const M2vTexture,
    usage: u32,
    error: *mut *mut c_char,
) -> bool {
    let Some(encoder) = encoder.as_mut().and_then(|e| e.0.as_mut()) else {
        return fail(error, "the encoder is NULL or ended", false);
    };
    let resource = check!(resource(buffer, texture), error, false);
    let usage = check!(raw::<ResourceUsage>(usage, "MTLResourceUsage"), error, false);
    check!(encoder.use_resource(resource, usage), error, false);
    true
}
