//! The render half of the C ABI: textures, samplers, depth and stencil states, render pipelines,
//! render passes, and render and blit command encoders. Descriptors cross as flat C structures
//! (`include/metal2vulkan.h`) and enums as Metal's raw values.
//!
//! SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later

use super::*;
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::metal::state::{DepthStencilDescriptor, StencilDescriptor};
use metal2vulkan::metal::{
    BlendFactor, BlendOperation, ColorWriteMask, CompareFunction, CullMode, IndexType, PixelFormat,
    PrimitiveTopologyClass, SamplerAddressMode, SamplerBorderColor, SamplerMinMagFilter,
    SamplerMipFilter, StencilOperation, TextureType, TextureUsage, TriangleFillMode, VertexFormat,
    VertexStepFunction, Winding,
};
use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource, VertexAttributeDescriptor,
    VertexBufferLayoutDescriptor, VertexDescriptor,
};
use metal2vulkan_runtime::{
    BlitCommandEncoder, BlitOption, ClearColor, DepthStencilState, DrawPrimitiveType, LoadAction,
    Origin, Region, RenderCommandEncoder, RenderPassDescriptor, RenderPipelineState, SamplerState,
    ScissorRect, StoreAction, Texture, TextureDescriptor, Viewport, VisibilityResultMode,
};

pub struct M2vTexture(Texture);
pub struct M2vSamplerState(SamplerState);
pub struct M2vDepthStencilState(DepthStencilState);
pub struct M2vRenderPipeline(pub(crate) RenderPipelineState);
pub struct M2vRenderEncoder(Option<RenderCommandEncoder>);
pub struct M2vBlitEncoder(Option<BlitCommandEncoder>);

/// A Metal enum from its raw value, or why not.
fn raw<T: TryFrom<u32>>(value: u32, what: &str) -> Result<T, String> {
    T::try_from(value).map_err(|_| format!("{value} is not a valid {what}"))
}

fn format(value: u32) -> Result<Option<PixelFormat>, String> {
    if value == 0 {
        Ok(None)
    } else {
        raw(value, "MTLPixelFormat").map(Some)
    }
}

impl From<M2vSize> for Origin {
    fn from(size: M2vSize) -> Origin {
        Origin::new(size.width, size.height, size.depth)
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct M2vRegion {
    origin: M2vSize,
    size: M2vSize,
}

impl From<M2vRegion> for Region {
    fn from(region: M2vRegion) -> Region {
        Region { origin: region.origin.into(), size: region.size.into() }
    }
}

// MARK: Textures

#[repr(C)]
pub struct M2vTextureDescriptor {
    texture_type: u32,
    pixel_format: u32,
    width: u64,
    height: u64,
    depth: u64,
    mipmap_level_count: u64,
    sample_count: u64,
    array_length: u64,
    usage: u64,
    resource_options: u64,
}

unsafe fn texture_descriptor(d: &M2vTextureDescriptor) -> Result<TextureDescriptor, String> {
    let mut descriptor = TextureDescriptor::default();
    descriptor.texture_type = raw::<TextureType>(d.texture_type, "MTLTextureType")?;
    descriptor.pixel_format = format(d.pixel_format)?.ok_or("MTLPixelFormatInvalid")?;
    descriptor.width = d.width;
    descriptor.height = d.height;
    descriptor.depth = d.depth;
    descriptor.mipmap_level_count = d.mipmap_level_count;
    descriptor.sample_count = d.sample_count;
    descriptor.array_length = d.array_length;
    descriptor.usage = raw::<TextureUsage>(d.usage as u32, "MTLTextureUsage")?;
    descriptor.resource_options = ResourceOptions::from_raw(d.resource_options).map_err(|e| e.to_string())?;
    Ok(descriptor)
}

#[no_mangle]
pub unsafe extern "C" fn m2v_device_new_texture(
    device: *const M2vDevice,
    descriptor: *const M2vTextureDescriptor,
    error: *mut *mut c_char,
) -> *mut M2vTexture {
    let device = handle!(device, error, ptr::null_mut());
    let descriptor = handle!(descriptor, error, ptr::null_mut());
    let descriptor = check!(texture_descriptor(descriptor), error, ptr::null_mut());
    boxed(M2vTexture(check!(device.device.new_texture(&descriptor), error, ptr::null_mut())))
}

#[no_mangle]
pub unsafe extern "C" fn m2v_texture_release(texture: *mut M2vTexture) {
    release(texture)
}

#[no_mangle]
pub unsafe extern "C" fn m2v_texture_descriptor_get(
    texture: *const M2vTexture,
    out: *mut M2vTextureDescriptor,
) {
    let (Some(texture), Some(out)) = (texture.as_ref(), out.as_mut()) else { return };
    let t = &texture.0;
    *out = M2vTextureDescriptor {
        texture_type: t.texture_type().as_raw(),
        pixel_format: t.pixel_format().as_raw(),
        width: t.width(),
        height: t.height(),
        depth: t.depth(),
        mipmap_level_count: t.mipmap_level_count(),
        sample_count: t.sample_count(),
        array_length: t.array_length(),
        usage: t.usage().as_raw() as u64,
        resource_options: t.resource_options().raw(),
    };
}

#[no_mangle]
pub unsafe extern "C" fn m2v_texture_replace_region(
    texture: *const M2vTexture,
    region: M2vRegion,
    level: u64,
    slice: u64,
    bytes: *const c_void,
    length: usize,
    bytes_per_row: u64,
    bytes_per_image: u64,
    error: *mut *mut c_char,
) -> bool {
    let texture = handle!(texture, error, false);
    if bytes.is_null() {
        return fail(error, "bytes is NULL", false);
    }
    let bytes = std::slice::from_raw_parts(bytes.cast::<u8>(), length);
    check!(
        texture.0.replace_region(region.into(), level, slice, bytes, bytes_per_row, bytes_per_image),
        error,
        false
    );
    true
}

#[no_mangle]
pub unsafe extern "C" fn m2v_texture_get_bytes(
    texture: *const M2vTexture,
    bytes: *mut c_void,
    length: usize,
    bytes_per_row: u64,
    bytes_per_image: u64,
    region: M2vRegion,
    level: u64,
    slice: u64,
    error: *mut *mut c_char,
) -> bool {
    let texture = handle!(texture, error, false);
    if bytes.is_null() {
        return fail(error, "bytes is NULL", false);
    }
    let bytes = std::slice::from_raw_parts_mut(bytes.cast::<u8>(), length);
    check!(
        texture.0.get_bytes(bytes, bytes_per_row, bytes_per_image, region.into(), level, slice),
        error,
        false
    );
    true
}

#[no_mangle]
pub unsafe extern "C" fn m2v_texture_new_view(
    texture: *const M2vTexture,
    pixel_format: u32,
    texture_type: u32,
    first_level: u64,
    level_count: u64,
    first_slice: u64,
    slice_count: u64,
    error: *mut *mut c_char,
) -> *mut M2vTexture {
    let texture = handle!(texture, error, ptr::null_mut());
    let pixel_format =
        check!(format(pixel_format).and_then(|f| f.ok_or("MTLPixelFormatInvalid".into())), error, ptr::null_mut());
    let texture_type = check!(raw::<TextureType>(texture_type, "MTLTextureType"), error, ptr::null_mut());
    let view = texture.0.new_texture_view_with_levels_slices(
        pixel_format,
        texture_type,
        first_level..first_level + level_count,
        first_slice..first_slice + slice_count,
    );
    boxed(M2vTexture(check!(view, error, ptr::null_mut())))
}

// MARK: Samplers

#[repr(C)]
pub struct M2vSamplerDescriptor {
    min_filter: u32,
    mag_filter: u32,
    mip_filter: u32,
    s_address_mode: u32,
    t_address_mode: u32,
    r_address_mode: u32,
    border_color: u32,
    normalized_coordinates: bool,
    lod_min_clamp: f32,
    lod_max_clamp: f32,
    max_anisotropy: u32,
    compare_function: u32,
}

#[no_mangle]
pub unsafe extern "C" fn m2v_device_new_sampler_state(
    device: *const M2vDevice,
    descriptor: *const M2vSamplerDescriptor,
    error: *mut *mut c_char,
) -> *mut M2vSamplerState {
    let device = handle!(device, error, ptr::null_mut());
    let d = handle!(descriptor, error, ptr::null_mut());
    let build = || -> Result<SamplerDescriptor, String> {
        let mut s = SamplerDescriptor::new();
        s.min_filter = raw::<SamplerMinMagFilter>(d.min_filter, "MTLSamplerMinMagFilter")?;
        s.mag_filter = raw::<SamplerMinMagFilter>(d.mag_filter, "MTLSamplerMinMagFilter")?;
        s.mip_filter = raw::<SamplerMipFilter>(d.mip_filter, "MTLSamplerMipFilter")?;
        s.s_address_mode = raw::<SamplerAddressMode>(d.s_address_mode, "MTLSamplerAddressMode")?;
        s.t_address_mode = raw::<SamplerAddressMode>(d.t_address_mode, "MTLSamplerAddressMode")?;
        s.r_address_mode = raw::<SamplerAddressMode>(d.r_address_mode, "MTLSamplerAddressMode")?;
        s.border_color = raw::<SamplerBorderColor>(d.border_color, "MTLSamplerBorderColor")?;
        s.normalized_coordinates = d.normalized_coordinates;
        s.lod_min_clamp = d.lod_min_clamp;
        s.lod_max_clamp = d.lod_max_clamp;
        s.max_anisotropy = d.max_anisotropy.max(1);
        s.compare_function = raw::<CompareFunction>(d.compare_function, "MTLCompareFunction")?;
        Ok(s)
    };
    let descriptor = check!(build(), error, ptr::null_mut());
    boxed(M2vSamplerState(check!(device.device.new_sampler_state(&descriptor), error, ptr::null_mut())))
}

#[no_mangle]
pub unsafe extern "C" fn m2v_sampler_state_release(sampler: *mut M2vSamplerState) {
    release(sampler)
}

// MARK: Depth and stencil

#[repr(C)]
pub struct M2vStencilDescriptor {
    compare_function: u32,
    stencil_failure_operation: u32,
    depth_failure_operation: u32,
    depth_stencil_pass_operation: u32,
    read_mask: u32,
    write_mask: u32,
}

#[repr(C)]
pub struct M2vDepthStencilDescriptor {
    depth_compare_function: u32,
    depth_write_enabled: bool,
    front_face_stencil: M2vStencilDescriptor,
    back_face_stencil: M2vStencilDescriptor,
}

fn stencil(d: &M2vStencilDescriptor) -> Result<StencilDescriptor, String> {
    let mut s = StencilDescriptor::default();
    s.stencil_compare_function = raw::<CompareFunction>(d.compare_function, "MTLCompareFunction")?;
    s.stencil_failure_operation = raw::<StencilOperation>(d.stencil_failure_operation, "MTLStencilOperation")?;
    s.depth_failure_operation = raw::<StencilOperation>(d.depth_failure_operation, "MTLStencilOperation")?;
    s.depth_stencil_pass_operation =
        raw::<StencilOperation>(d.depth_stencil_pass_operation, "MTLStencilOperation")?;
    s.read_mask = d.read_mask;
    s.write_mask = d.write_mask;
    Ok(s)
}

#[no_mangle]
pub unsafe extern "C" fn m2v_device_new_depth_stencil_state(
    device: *const M2vDevice,
    descriptor: *const M2vDepthStencilDescriptor,
    error: *mut *mut c_char,
) -> *mut M2vDepthStencilState {
    let device = handle!(device, error, ptr::null_mut());
    let d = handle!(descriptor, error, ptr::null_mut());
    let build = || -> Result<DepthStencilDescriptor, String> {
        let mut s = DepthStencilDescriptor::default();
        s.depth_compare_function = raw::<CompareFunction>(d.depth_compare_function, "MTLCompareFunction")?;
        s.depth_write_enabled = d.depth_write_enabled;
        s.front_face_stencil = stencil(&d.front_face_stencil)?;
        s.back_face_stencil = stencil(&d.back_face_stencil)?;
        Ok(s)
    };
    let descriptor = check!(build(), error, ptr::null_mut());
    let state = device.device.new_depth_stencil_state(&descriptor);
    boxed(M2vDepthStencilState(check!(state, error, ptr::null_mut())))
}

#[no_mangle]
pub unsafe extern "C" fn m2v_depth_stencil_state_release(state: *mut M2vDepthStencilState) {
    release(state)
}

// MARK: Render pipelines

#[repr(C)]
pub struct M2vColorAttachment {
    /// 0 for none.
    pixel_format: u32,
    blending_enabled: bool,
    source_rgb_blend_factor: u32,
    destination_rgb_blend_factor: u32,
    rgb_blend_operation: u32,
    source_alpha_blend_factor: u32,
    destination_alpha_blend_factor: u32,
    alpha_blend_operation: u32,
    write_mask: u32,
}

#[repr(C)]
pub struct M2vVertexAttribute {
    /// 0 (MTLVertexFormatInvalid) for none.
    format: u32,
    offset: u32,
    buffer_index: u32,
}

#[repr(C)]
pub struct M2vVertexLayout {
    /// UINT32_MAX for MTLBufferLayoutStrideDynamic.
    stride: u32,
    step_function: u32,
    step_rate: u32,
}

pub const VERTEX_SLOTS: usize = 31;

#[repr(C)]
pub struct M2vRenderPipelineDescriptor {
    vertex_function: *const M2vFunction,
    fragment_function: *const M2vFunction,
    color_attachments: [M2vColorAttachment; 8],
    depth_attachment_pixel_format: u32,
    stencil_attachment_pixel_format: u32,
    raster_sample_count: u32,
    alpha_to_coverage_enabled: bool,
    alpha_to_one_enabled: bool,
    rasterization_enabled: bool,
    input_primitive_topology: u32,
    /// Whether the pipeline has a vertex descriptor.
    has_vertex_descriptor: bool,
    attributes: [M2vVertexAttribute; VERTEX_SLOTS],
    layouts: [M2vVertexLayout; VERTEX_SLOTS],
}

unsafe fn pipeline_descriptor(d: &M2vRenderPipelineDescriptor) -> Result<RenderPipelineDescriptor, String> {
    let vertex = d.vertex_function.as_ref().ok_or("no vertex function")?;
    let mut p = RenderPipelineDescriptor::new(StageSource::Function(vertex.function.function_descriptor()));
    if let Some(fragment) = d.fragment_function.as_ref() {
        p.fragment = Some(StageSource::Function(fragment.function.function_descriptor()));
    }
    for (i, c) in d.color_attachments.iter().enumerate() {
        let Some(pixel_format) = format(c.pixel_format)? else { continue };
        let mut a = ColorAttachmentDescriptor::new(pixel_format);
        a.blending_enabled = c.blending_enabled;
        a.source_rgb_blend_factor = raw::<BlendFactor>(c.source_rgb_blend_factor, "MTLBlendFactor")?;
        a.destination_rgb_blend_factor = raw::<BlendFactor>(c.destination_rgb_blend_factor, "MTLBlendFactor")?;
        a.rgb_blend_operation = raw::<BlendOperation>(c.rgb_blend_operation, "MTLBlendOperation")?;
        a.source_alpha_blend_factor = raw::<BlendFactor>(c.source_alpha_blend_factor, "MTLBlendFactor")?;
        a.destination_alpha_blend_factor =
            raw::<BlendFactor>(c.destination_alpha_blend_factor, "MTLBlendFactor")?;
        a.alpha_blend_operation = raw::<BlendOperation>(c.alpha_blend_operation, "MTLBlendOperation")?;
        a.write_mask = raw::<ColorWriteMask>(c.write_mask, "MTLColorWriteMask")?;
        p.color_attachments[i] = Some(a);
    }
    p.depth_format = format(d.depth_attachment_pixel_format)?;
    p.stencil_format = format(d.stencil_attachment_pixel_format)?;
    p.raster_sample_count = d.raster_sample_count.max(1);
    p.alpha_to_coverage = d.alpha_to_coverage_enabled;
    p.alpha_to_one = d.alpha_to_one_enabled;
    p.rasterization_enabled = d.rasterization_enabled;
    p.input_primitive_topology =
        raw::<PrimitiveTopologyClass>(d.input_primitive_topology, "MTLPrimitiveTopologyClass")?;
    if d.has_vertex_descriptor {
        let mut v = VertexDescriptor::new();
        for (i, a) in d.attributes.iter().enumerate() {
            let vertex_format = raw::<VertexFormat>(a.format, "MTLVertexFormat")?;
            if a.format == 0 {
                continue;
            }
            v.attributes.insert(i as u32, VertexAttributeDescriptor::new(vertex_format, a.offset, a.buffer_index));
            // A layout for each buffer an attribute reads, as Metal's are all present.
            let index = a.buffer_index as usize;
            if index < VERTEX_SLOTS && !v.layouts.contains_key(&a.buffer_index) {
                let l = &d.layouts[index];
                let mut layout =
                    VertexBufferLayoutDescriptor::new(if l.stride == u32::MAX { None } else { Some(l.stride) });
                layout.step_function = raw::<VertexStepFunction>(l.step_function, "MTLVertexStepFunction")?;
                layout.step_rate = l.step_rate;
                v.layouts.insert(a.buffer_index, layout);
            }
        }
        p.vertex_descriptor = Some(v);
    }
    Ok(p)
}

#[no_mangle]
pub unsafe extern "C" fn m2v_device_new_render_pipeline_state(
    device: *const M2vDevice,
    descriptor: *const M2vRenderPipelineDescriptor,
    error: *mut *mut c_char,
) -> *mut M2vRenderPipeline {
    let device = handle!(device, error, ptr::null_mut());
    let descriptor = handle!(descriptor, error, ptr::null_mut());
    let descriptor = check!(pipeline_descriptor(descriptor), error, ptr::null_mut());
    let state = device.device.new_render_pipeline_state(&descriptor, &PipelineOptions::default());
    boxed(M2vRenderPipeline(check!(state, error, ptr::null_mut())))
}

#[no_mangle]
pub unsafe extern "C" fn m2v_render_pipeline_release(pipeline: *mut M2vRenderPipeline) {
    release(pipeline)
}

// MARK: Render passes

#[repr(C)]
pub struct M2vPassAttachment {
    texture: *const M2vTexture,
    level: u64,
    slice: u64,
    depth_plane: u64,
    resolve_texture: *const M2vTexture,
    resolve_level: u64,
    resolve_slice: u64,
    resolve_depth_plane: u64,
    load_action: u64,
    store_action: u64,
    /// Red, green, blue, alpha for a colour; depth in [0]; stencil in [0] as a number.
    clear: [f64; 4],
}

#[repr(C)]
pub struct M2vRenderPassDescriptor {
    color_attachments: [M2vPassAttachment; 8],
    depth_attachment: M2vPassAttachment,
    stencil_attachment: M2vPassAttachment,
    visibility_result_buffer: *const M2vBuffer,
    render_target_width: u64,
    render_target_height: u64,
    render_target_array_length: u64,
    default_raster_sample_count: u64,
}

fn texture_of(texture: *const M2vTexture) -> Option<Texture> {
    unsafe { texture.as_ref() }.map(|t| t.0.clone())
}

unsafe fn pass_descriptor(d: &M2vRenderPassDescriptor) -> Result<RenderPassDescriptor, String> {
    let mut pass = RenderPassDescriptor::new();
    let load = |raw: u64| LoadAction::from_raw(raw).map_err(|e| e.to_string());
    let store = |raw: u64| StoreAction::from_raw(raw).map_err(|e| e.to_string());
    for (i, a) in d.color_attachments.iter().enumerate() {
        let c = &mut pass.color_attachments[i];
        c.texture = texture_of(a.texture);
        c.level = a.level;
        c.slice = a.slice;
        c.depth_plane = a.depth_plane;
        c.resolve_texture = texture_of(a.resolve_texture);
        c.resolve_level = a.resolve_level;
        c.resolve_slice = a.resolve_slice;
        c.resolve_depth_plane = a.resolve_depth_plane;
        c.load_action = load(a.load_action)?;
        c.store_action = store(a.store_action)?;
        c.clear_color = ClearColor::new(a.clear[0], a.clear[1], a.clear[2], a.clear[3]);
    }
    let a = &d.depth_attachment;
    let depth = &mut pass.depth_attachment;
    depth.texture = texture_of(a.texture);
    depth.level = a.level;
    depth.slice = a.slice;
    depth.depth_plane = a.depth_plane;
    depth.resolve_texture = texture_of(a.resolve_texture);
    depth.load_action = load(a.load_action)?;
    depth.store_action = store(a.store_action)?;
    depth.clear_depth = a.clear[0];
    let a = &d.stencil_attachment;
    let stencil = &mut pass.stencil_attachment;
    stencil.texture = texture_of(a.texture);
    stencil.level = a.level;
    stencil.slice = a.slice;
    stencil.depth_plane = a.depth_plane;
    stencil.resolve_texture = texture_of(a.resolve_texture);
    stencil.load_action = load(a.load_action)?;
    stencil.store_action = store(a.store_action)?;
    stencil.clear_stencil = a.clear[0] as u32;
    pass.visibility_result_buffer = d.visibility_result_buffer.as_ref().map(|b| b.0.clone());
    pass.render_target_width = d.render_target_width;
    pass.render_target_height = d.render_target_height;
    pass.render_target_array_length = d.render_target_array_length;
    pass.default_raster_sample_count = d.default_raster_sample_count;
    Ok(pass)
}

#[no_mangle]
pub unsafe extern "C" fn m2v_command_buffer_render_command_encoder(
    buffer: *const M2vCommandBuffer,
    descriptor: *const M2vRenderPassDescriptor,
    error: *mut *mut c_char,
) -> *mut M2vRenderEncoder {
    let buffer = handle!(buffer, error, ptr::null_mut());
    let descriptor = handle!(descriptor, error, ptr::null_mut());
    let pass = check!(pass_descriptor(descriptor), error, ptr::null_mut());
    let encoder = check!(buffer.0.render_command_encoder(&pass), error, ptr::null_mut());
    boxed(M2vRenderEncoder(Some(encoder)))
}

#[no_mangle]
pub unsafe extern "C" fn m2v_render_encoder_release(encoder: *mut M2vRenderEncoder) {
    release(encoder)
}

macro_rules! render {
    ($encoder:expr, $error:expr) => {
        match $encoder.as_mut().and_then(|encoder| encoder.0.as_mut()) {
            Some(encoder) => encoder,
            None => return fail($error, "the encoder is NULL or ended", false),
        }
    };
}

/// The render encoder's binding calls for one stage.
macro_rules! stage_calls {
    ($buffer:ident, $offset:ident, $bytes:ident, $texture:ident, $sampler:ident,
     $set_buffer:ident, $set_offset:ident, $set_bytes:ident, $set_texture:ident, $set_sampler:ident) => {
        #[no_mangle]
        pub unsafe extern "C" fn $buffer(
            encoder: *mut M2vRenderEncoder,
            buffer: *const M2vBuffer,
            offset: u64,
            index: u32,
            error: *mut *mut c_char,
        ) -> bool {
            let encoder = render!(encoder, error);
            check!(encoder.$set_buffer(buffer.as_ref().map(|b| &b.0), offset, index), error, false);
            true
        }

        #[no_mangle]
        pub unsafe extern "C" fn $offset(
            encoder: *mut M2vRenderEncoder,
            offset: u64,
            index: u32,
            error: *mut *mut c_char,
        ) -> bool {
            let encoder = render!(encoder, error);
            check!(encoder.$set_offset(offset, index), error, false);
            true
        }

        #[no_mangle]
        pub unsafe extern "C" fn $bytes(
            encoder: *mut M2vRenderEncoder,
            bytes: *const c_void,
            length: usize,
            index: u32,
            error: *mut *mut c_char,
        ) -> bool {
            let encoder = render!(encoder, error);
            if bytes.is_null() {
                return fail(error, "bytes is NULL", false);
            }
            let bytes = std::slice::from_raw_parts(bytes.cast::<u8>(), length);
            check!(encoder.$set_bytes(bytes, index), error, false);
            true
        }

        #[no_mangle]
        pub unsafe extern "C" fn $texture(
            encoder: *mut M2vRenderEncoder,
            texture: *const M2vTexture,
            index: u32,
            error: *mut *mut c_char,
        ) -> bool {
            let encoder = render!(encoder, error);
            check!(encoder.$set_texture(texture.as_ref().map(|t| &t.0), index), error, false);
            true
        }

        #[no_mangle]
        pub unsafe extern "C" fn $sampler(
            encoder: *mut M2vRenderEncoder,
            sampler: *const M2vSamplerState,
            index: u32,
            error: *mut *mut c_char,
        ) -> bool {
            let encoder = render!(encoder, error);
            check!(encoder.$set_sampler(sampler.as_ref().map(|s| &s.0), index), error, false);
            true
        }
    };
}

stage_calls!(
    m2v_render_encoder_set_vertex_buffer,
    m2v_render_encoder_set_vertex_buffer_offset,
    m2v_render_encoder_set_vertex_bytes,
    m2v_render_encoder_set_vertex_texture,
    m2v_render_encoder_set_vertex_sampler_state,
    set_vertex_buffer,
    set_vertex_buffer_offset,
    set_vertex_bytes,
    set_vertex_texture,
    set_vertex_sampler_state
);
stage_calls!(
    m2v_render_encoder_set_fragment_buffer,
    m2v_render_encoder_set_fragment_buffer_offset,
    m2v_render_encoder_set_fragment_bytes,
    m2v_render_encoder_set_fragment_texture,
    m2v_render_encoder_set_fragment_sampler_state,
    set_fragment_buffer,
    set_fragment_buffer_offset,
    set_fragment_bytes,
    set_fragment_texture,
    set_fragment_sampler_state
);

#[no_mangle]
pub unsafe extern "C" fn m2v_render_encoder_set_render_pipeline_state(
    encoder: *mut M2vRenderEncoder,
    pipeline: *const M2vRenderPipeline,
    error: *mut *mut c_char,
) -> bool {
    let encoder = render!(encoder, error);
    let pipeline = handle!(pipeline, error, false);
    check!(encoder.set_render_pipeline_state(&pipeline.0), error, false);
    true
}

/// MTLViewport.
#[repr(C)]
pub struct M2vViewport {
    origin_x: f64,
    origin_y: f64,
    width: f64,
    height: f64,
    znear: f64,
    zfar: f64,
}

#[no_mangle]
pub unsafe extern "C" fn m2v_render_encoder_set_viewport(
    encoder: *mut M2vRenderEncoder,
    v: M2vViewport,
    error: *mut *mut c_char,
) -> bool {
    let encoder = render!(encoder, error);
    let viewport = Viewport::new(v.origin_x, v.origin_y, v.width, v.height, v.znear, v.zfar);
    check!(encoder.set_viewport(viewport), error, false);
    true
}

#[no_mangle]
pub unsafe extern "C" fn m2v_render_encoder_set_scissor_rect(
    encoder: *mut M2vRenderEncoder,
    x: u64,
    y: u64,
    width: u64,
    height: u64,
    error: *mut *mut c_char,
) -> bool {
    let encoder = render!(encoder, error);
    check!(encoder.set_scissor_rect(ScissorRect::new(x, y, width, height)), error, false);
    true
}

#[no_mangle]
pub unsafe extern "C" fn m2v_render_encoder_set_cull_mode(
    encoder: *mut M2vRenderEncoder,
    mode: u32,
    error: *mut *mut c_char,
) -> bool {
    let encoder = render!(encoder, error);
    encoder.set_cull_mode(check!(raw::<CullMode>(mode, "MTLCullMode"), error, false));
    true
}

#[no_mangle]
pub unsafe extern "C" fn m2v_render_encoder_set_front_facing_winding(
    encoder: *mut M2vRenderEncoder,
    winding: u32,
    error: *mut *mut c_char,
) -> bool {
    let encoder = render!(encoder, error);
    encoder.set_front_facing_winding(check!(raw::<Winding>(winding, "MTLWinding"), error, false));
    true
}

#[no_mangle]
pub unsafe extern "C" fn m2v_render_encoder_set_triangle_fill_mode(
    encoder: *mut M2vRenderEncoder,
    mode: u32,
    error: *mut *mut c_char,
) -> bool {
    let encoder = render!(encoder, error);
    let mode = check!(raw::<TriangleFillMode>(mode, "MTLTriangleFillMode"), error, false);
    check!(encoder.set_triangle_fill_mode(mode), error, false);
    true
}

#[no_mangle]
pub unsafe extern "C" fn m2v_render_encoder_set_depth_stencil_state(
    encoder: *mut M2vRenderEncoder,
    state: *const M2vDepthStencilState,
    error: *mut *mut c_char,
) -> bool {
    let encoder = render!(encoder, error);
    check!(encoder.set_depth_stencil_state(state.as_ref().map(|s| &s.0)), error, false);
    true
}

#[no_mangle]
pub unsafe extern "C" fn m2v_render_encoder_set_stencil_reference_values(
    encoder: *mut M2vRenderEncoder,
    front: u32,
    back: u32,
    error: *mut *mut c_char,
) -> bool {
    let encoder = render!(encoder, error);
    encoder.set_stencil_reference_values(front, back);
    true
}

#[no_mangle]
pub unsafe extern "C" fn m2v_render_encoder_set_blend_color(
    encoder: *mut M2vRenderEncoder,
    red: f32,
    green: f32,
    blue: f32,
    alpha: f32,
    error: *mut *mut c_char,
) -> bool {
    let encoder = render!(encoder, error);
    encoder.set_blend_color(red, green, blue, alpha);
    true
}

#[no_mangle]
pub unsafe extern "C" fn m2v_render_encoder_set_depth_bias(
    encoder: *mut M2vRenderEncoder,
    depth_bias: f32,
    slope_scale: f32,
    clamp: f32,
    error: *mut *mut c_char,
) -> bool {
    let encoder = render!(encoder, error);
    check!(encoder.set_depth_bias(depth_bias, slope_scale, clamp), error, false);
    true
}

#[no_mangle]
pub unsafe extern "C" fn m2v_render_encoder_set_visibility_result_mode(
    encoder: *mut M2vRenderEncoder,
    mode: u64,
    offset: u64,
    error: *mut *mut c_char,
) -> bool {
    let encoder = render!(encoder, error);
    let mode = check!(VisibilityResultMode::from_raw(mode), error, false);
    check!(encoder.set_visibility_result_mode(mode, offset), error, false);
    true
}

#[no_mangle]
pub unsafe extern "C" fn m2v_render_encoder_draw_primitives(
    encoder: *mut M2vRenderEncoder,
    primitive_type: u64,
    vertex_start: u32,
    vertex_count: u32,
    instance_count: u32,
    base_instance: u32,
    error: *mut *mut c_char,
) -> bool {
    let encoder = render!(encoder, error);
    let primitive = check!(DrawPrimitiveType::from_raw(primitive_type), error, false);
    check!(
        encoder.draw_primitives(primitive, vertex_start, vertex_count, instance_count, base_instance),
        error,
        false
    );
    true
}

#[no_mangle]
pub unsafe extern "C" fn m2v_render_encoder_draw_indexed_primitives(
    encoder: *mut M2vRenderEncoder,
    primitive_type: u64,
    index_count: u32,
    index_type: u32,
    index_buffer: *const M2vBuffer,
    index_buffer_offset: u64,
    instance_count: u32,
    base_vertex: i32,
    base_instance: u32,
    error: *mut *mut c_char,
) -> bool {
    let encoder = render!(encoder, error);
    let primitive = check!(DrawPrimitiveType::from_raw(primitive_type), error, false);
    let index_type = check!(raw::<IndexType>(index_type, "MTLIndexType"), error, false);
    let index_buffer = handle!(index_buffer, error, false);
    check!(
        encoder.draw_indexed_primitives(
            primitive,
            index_count,
            index_type,
            &index_buffer.0,
            index_buffer_offset,
            instance_count,
            base_vertex,
            base_instance
        ),
        error,
        false
    );
    true
}

#[no_mangle]
pub unsafe extern "C" fn m2v_render_encoder_draw_primitives_indirect(
    encoder: *mut M2vRenderEncoder,
    primitive_type: u64,
    indirect_buffer: *const M2vBuffer,
    indirect_buffer_offset: u64,
    error: *mut *mut c_char,
) -> bool {
    let encoder = render!(encoder, error);
    let primitive = check!(DrawPrimitiveType::from_raw(primitive_type), error, false);
    let indirect = handle!(indirect_buffer, error, false);
    check!(encoder.draw_primitives_indirect(primitive, &indirect.0, indirect_buffer_offset), error, false);
    true
}

#[no_mangle]
pub unsafe extern "C" fn m2v_render_encoder_draw_indexed_primitives_indirect(
    encoder: *mut M2vRenderEncoder,
    primitive_type: u64,
    index_type: u32,
    index_buffer: *const M2vBuffer,
    index_buffer_offset: u64,
    indirect_buffer: *const M2vBuffer,
    indirect_buffer_offset: u64,
    error: *mut *mut c_char,
) -> bool {
    let encoder = render!(encoder, error);
    let primitive = check!(DrawPrimitiveType::from_raw(primitive_type), error, false);
    let index_type = check!(raw::<IndexType>(index_type, "MTLIndexType"), error, false);
    let index_buffer = handle!(index_buffer, error, false);
    let indirect = handle!(indirect_buffer, error, false);
    check!(
        encoder.draw_indexed_primitives_indirect(
            primitive,
            index_type,
            &index_buffer.0,
            index_buffer_offset,
            &indirect.0,
            indirect_buffer_offset
        ),
        error,
        false
    );
    true
}

#[no_mangle]
pub unsafe extern "C" fn m2v_render_encoder_end(encoder: *mut M2vRenderEncoder, error: *mut *mut c_char) -> bool {
    if encoder.is_null() {
        return fail(error, "the encoder is NULL", false);
    }
    let encoder = Box::from_raw(encoder);
    let Some(encoder) = encoder.0 else { return fail(error, "the encoder already ended", false) };
    check!(encoder.end_encoding(), error, false);
    true
}

// MARK: Blit encoders

#[no_mangle]
pub unsafe extern "C" fn m2v_command_buffer_blit_command_encoder(
    buffer: *const M2vCommandBuffer,
    error: *mut *mut c_char,
) -> *mut M2vBlitEncoder {
    let buffer = handle!(buffer, error, ptr::null_mut());
    boxed(M2vBlitEncoder(Some(check!(buffer.0.blit_command_encoder(), error, ptr::null_mut()))))
}

#[no_mangle]
pub unsafe extern "C" fn m2v_blit_encoder_release(encoder: *mut M2vBlitEncoder) {
    release(encoder)
}

macro_rules! blit {
    ($encoder:expr, $error:expr) => {
        match $encoder.as_mut().and_then(|encoder| encoder.0.as_mut()) {
            Some(encoder) => encoder,
            None => return fail($error, "the encoder is NULL or ended", false),
        }
    };
}

fn blit_option(raw: u64) -> Result<BlitOption, String> {
    match raw {
        0 => Ok(BlitOption::None),
        1 => Ok(BlitOption::DepthFromDepthStencil),
        2 => Ok(BlitOption::StencilFromDepthStencil),
        4 => Ok(BlitOption::RowLinearPVRTC),
        _ => Err(format!("{raw} is not a valid MTLBlitOption")),
    }
}

#[no_mangle]
pub unsafe extern "C" fn m2v_blit_encoder_copy_buffer_to_buffer(
    encoder: *mut M2vBlitEncoder,
    source: *const M2vBuffer,
    source_offset: u64,
    destination: *const M2vBuffer,
    destination_offset: u64,
    size: u64,
    error: *mut *mut c_char,
) -> bool {
    let encoder = blit!(encoder, error);
    let source = handle!(source, error, false);
    let destination = handle!(destination, error, false);
    check!(
        encoder.copy_from_buffer_to_buffer(&source.0, source_offset, &destination.0, destination_offset, size),
        error,
        false
    );
    true
}

#[no_mangle]
pub unsafe extern "C" fn m2v_blit_encoder_copy_buffer_to_texture(
    encoder: *mut M2vBlitEncoder,
    source: *const M2vBuffer,
    source_offset: u64,
    source_bytes_per_row: u64,
    source_bytes_per_image: u64,
    source_size: M2vSize,
    destination: *const M2vTexture,
    destination_slice: u64,
    destination_level: u64,
    destination_origin: M2vSize,
    options: u64,
    error: *mut *mut c_char,
) -> bool {
    let encoder = blit!(encoder, error);
    let source = handle!(source, error, false);
    let destination = handle!(destination, error, false);
    let options = check!(blit_option(options), error, false);
    check!(
        encoder.copy_from_buffer_to_texture(
            &source.0,
            source_offset,
            source_bytes_per_row,
            source_bytes_per_image,
            source_size.into(),
            &destination.0,
            destination_slice,
            destination_level,
            destination_origin.into(),
            options
        ),
        error,
        false
    );
    true
}

#[no_mangle]
pub unsafe extern "C" fn m2v_blit_encoder_copy_texture_to_buffer(
    encoder: *mut M2vBlitEncoder,
    source: *const M2vTexture,
    source_slice: u64,
    source_level: u64,
    source_origin: M2vSize,
    source_size: M2vSize,
    destination: *const M2vBuffer,
    destination_offset: u64,
    destination_bytes_per_row: u64,
    destination_bytes_per_image: u64,
    options: u64,
    error: *mut *mut c_char,
) -> bool {
    let encoder = blit!(encoder, error);
    let source = handle!(source, error, false);
    let destination = handle!(destination, error, false);
    let options = check!(blit_option(options), error, false);
    check!(
        encoder.copy_from_texture_to_buffer(
            &source.0,
            source_slice,
            source_level,
            source_origin.into(),
            source_size.into(),
            &destination.0,
            destination_offset,
            destination_bytes_per_row,
            destination_bytes_per_image,
            options
        ),
        error,
        false
    );
    true
}

#[no_mangle]
pub unsafe extern "C" fn m2v_blit_encoder_copy_texture_to_texture(
    encoder: *mut M2vBlitEncoder,
    source: *const M2vTexture,
    source_slice: u64,
    source_level: u64,
    source_origin: M2vSize,
    source_size: M2vSize,
    destination: *const M2vTexture,
    destination_slice: u64,
    destination_level: u64,
    destination_origin: M2vSize,
    error: *mut *mut c_char,
) -> bool {
    let encoder = blit!(encoder, error);
    let source = handle!(source, error, false);
    let destination = handle!(destination, error, false);
    check!(
        encoder.copy_from_texture_to_texture(
            &source.0,
            source_slice,
            source_level,
            source_origin.into(),
            source_size.into(),
            &destination.0,
            destination_slice,
            destination_level,
            destination_origin.into()
        ),
        error,
        false
    );
    true
}

#[no_mangle]
pub unsafe extern "C" fn m2v_blit_encoder_generate_mipmaps(
    encoder: *mut M2vBlitEncoder,
    texture: *const M2vTexture,
    error: *mut *mut c_char,
) -> bool {
    let encoder = blit!(encoder, error);
    let texture = handle!(texture, error, false);
    check!(encoder.generate_mipmaps(&texture.0), error, false);
    true
}

#[no_mangle]
pub unsafe extern "C" fn m2v_blit_encoder_fill_buffer(
    encoder: *mut M2vBlitEncoder,
    buffer: *const M2vBuffer,
    location: u64,
    length: u64,
    value: u8,
    error: *mut *mut c_char,
) -> bool {
    let encoder = blit!(encoder, error);
    let buffer = handle!(buffer, error, false);
    check!(encoder.fill_buffer(&buffer.0, location..location + length, value), error, false);
    true
}

/// synchronizeResource: a buffer or, when `buffer` is NULL, a texture.
#[no_mangle]
pub unsafe extern "C" fn m2v_blit_encoder_synchronize_resource(
    encoder: *mut M2vBlitEncoder,
    buffer: *const M2vBuffer,
    texture: *const M2vTexture,
    error: *mut *mut c_char,
) -> bool {
    let encoder = blit!(encoder, error);
    let result = match (buffer.as_ref(), texture.as_ref()) {
        (Some(buffer), _) => encoder.synchronize_resource((&buffer.0).into()),
        (None, Some(texture)) => encoder.synchronize_resource((&texture.0).into()),
        (None, None) => return fail(error, "no resource", false),
    };
    check!(result, error, false);
    true
}

#[no_mangle]
pub unsafe extern "C" fn m2v_blit_encoder_end(encoder: *mut M2vBlitEncoder, error: *mut *mut c_char) -> bool {
    if encoder.is_null() {
        return fail(error, "the encoder is NULL", false);
    }
    let encoder = Box::from_raw(encoder);
    let Some(encoder) = encoder.0 else { return fail(error, "the encoder already ended", false) };
    check!(encoder.end_encoding(), error, false);
    true
}

// MARK: Reflection

/// MTLVertexAttribute.
#[repr(C)]
pub struct M2vAttribute {
    name: *mut c_char,
    attribute_index: u32,
    /// MTLDataType.
    attribute_type: u32,
    active: bool,
    patch_data: bool,
    patch_control_point_data: bool,
}

/// vertexAttributes: how many, stored as a malloc'd array in *attributes that
/// m2v_attributes_free frees; -1 for nil (not a vertex function, or the
/// runtime cannot say).
#[no_mangle]
pub unsafe extern "C" fn m2v_function_vertex_attributes(
    function: *const M2vFunction,
    attributes: *mut *mut M2vAttribute,
) -> i64 {
    let Some(function) = function.as_ref() else { return -1 };
    let Ok(Some(list)) = function.function.vertex_attributes() else { return -1 };
    let out: Vec<M2vAttribute> = list
        .iter()
        .map(|a| M2vAttribute {
            name: c_string(a.name()),
            attribute_index: a.attribute_index(),
            attribute_type: a.attribute_type().as_raw(),
            active: a.is_active(),
            patch_data: a.is_patch_data(),
            patch_control_point_data: a.is_patch_control_point_data(),
        })
        .collect();
    let count = out.len() as i64;
    if attributes.is_null() {
        m2v_attributes_free(Box::into_raw(out.into_boxed_slice()).cast(), count as usize);
    } else {
        *attributes = Box::into_raw(out.into_boxed_slice()).cast();
    }
    count
}

#[no_mangle]
pub unsafe extern "C" fn m2v_attributes_free(attributes: *mut M2vAttribute, count: usize) {
    if attributes.is_null() {
        return;
    }
    let list = Box::from_raw(ptr::slice_from_raw_parts_mut(attributes, count));
    for a in list.iter() {
        m2v_string_free(a.name);
    }
}

// MARK: Compute textures and samplers

/// memoryBarrierWithScope: (MTLBarrierScope).
#[no_mangle]
pub unsafe extern "C" fn m2v_compute_encoder_memory_barrier(
    encoder: *mut M2vComputeEncoder,
    scope: u64,
    error: *mut *mut c_char,
) -> bool {
    let Some(encoder) = encoder.as_mut().and_then(|e| e.0.as_mut()) else {
        return fail(error, "the encoder is NULL or ended", false);
    };
    let scope = check!(raw::<metal2vulkan::metal::BarrierScope>(scope as u32, "MTLBarrierScope"), error, false);
    encoder.memory_barrier_with_scope(scope);
    true
}

#[no_mangle]
pub unsafe extern "C" fn m2v_compute_encoder_set_texture(
    encoder: *mut M2vComputeEncoder,
    texture: *const M2vTexture,
    index: u32,
    error: *mut *mut c_char,
) -> bool {
    let Some(encoder) = encoder.as_mut().and_then(|e| e.0.as_mut()) else {
        return fail(error, "the encoder is NULL or ended", false);
    };
    check!(encoder.set_texture(texture.as_ref().map(|t| &t.0), index), error, false);
    true
}

#[no_mangle]
pub unsafe extern "C" fn m2v_compute_encoder_set_sampler_state(
    encoder: *mut M2vComputeEncoder,
    sampler: *const M2vSamplerState,
    index: u32,
    error: *mut *mut c_char,
) -> bool {
    let Some(encoder) = encoder.as_mut().and_then(|e| e.0.as_mut()) else {
        return fail(error, "the encoder is NULL or ended", false);
    };
    check!(encoder.set_sampler_state(sampler.as_ref().map(|s| &s.0), index), error, false);
    true
}
