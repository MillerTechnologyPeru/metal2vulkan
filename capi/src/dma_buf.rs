//! Textures over dma-bufs (ABI 11): a texture whose memory another API or process imports, such
//! as a drawable's that a compositor shows without the frame being read back. No Metal
//! counterpart: Core Animation shares a drawable's IOSurface instead.
//!
//! The texture is a linear (`DRM_FORMAT_MOD_LINEAR`) image of the device, made for dma-bufs and
//! bound to a dedicated allocation exported as one, wrapped by the runtime
//! (`Device::new_texture_with_vk_image`). It is made `UNDEFINED` and rests in `GENERAL`. The
//! image and its memory are the handle's, freed after its texture.
//!
//! SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later

use super::render::{texture_descriptor, M2vTexture, M2vTextureDescriptor};
use super::*;
use ash::vk;
use metal2vulkan_runtime::{StorageMode, VkImageLayouts};

/// A texture's image and memory, which the shim made and frees once the texture has gone.
pub(crate) struct DmaBufImage {
    vulkan: ash::Device,
    image: vk::Image,
    memory: vk::DeviceMemory,
    /// Keeps the device alive until the image and memory are freed.
    _device: Device,
}

impl Drop for DmaBufImage {
    fn drop(&mut self) {
        // SAFETY: the image and memory are this handle's, made on `vulkan`, which `_device` keeps
        // alive; the texture over the image went first (`M2vTexture`'s field order).
        unsafe {
            self.vulkan.destroy_image(self.image, None);
            self.vulkan.free_memory(self.memory, None);
        }
    }
}

/// The Vulkan format and DRM fourcc of the pixel formats a dma-buf texture takes.
fn formats(pixel_format: u32) -> Option<(vk::Format, u32)> {
    const AB24: u32 = u32::from_le_bytes(*b"AB24"); // DRM_FORMAT_ABGR8888: R, G, B, A in memory
    const AR24: u32 = u32::from_le_bytes(*b"AR24"); // DRM_FORMAT_ARGB8888: B, G, R, A in memory
    const AB30: u32 = u32::from_le_bytes(*b"AB30"); // DRM_FORMAT_ABGR2101010: R in the low 10 bits
    const AR30: u32 = u32::from_le_bytes(*b"AR30"); // DRM_FORMAT_ARGB2101010: B in the low 10 bits
    match pixel_format {
        70 => Some((vk::Format::R8G8B8A8_UNORM, AB24)),
        71 => Some((vk::Format::R8G8B8A8_SRGB, AB24)),
        80 => Some((vk::Format::B8G8R8A8_UNORM, AR24)),
        81 => Some((vk::Format::B8G8R8A8_SRGB, AR24)),
        90 => Some((vk::Format::A2B10G10R10_UNORM_PACK32, AB30)), // RGB10A2Unorm
        94 => Some((vk::Format::A2R10G10B10_UNORM_PACK32, AR30)), // BGR10A2Unorm
        _ => None,
    }
}

/// A 2D texture of `descriptor` (BGRA8 or RGBA8, unorm or sRGB, or RGB10A2 or BGR10A2 unorm; one
/// level and sample, private storage) over memory exported as a dma-buf, described in `dma_buf`:
/// its fd, which the caller owns and closes, then width, height, row stride, offset, DRM fourcc
/// and DRM format modifier.
/// NULL, with `error` set, where the device doesn't export dma-bufs as linear images.
///
/// The caller releases the texture only once every command buffer that used it has completed.
unsafe fn dma_buf_texture(
    device: &Device,
    descriptor: &M2vTextureDescriptor,
    dma_buf: &mut [i64; 7],
) -> Result<M2vTexture, String> {
    let metal = texture_descriptor(descriptor)?;
    let (format, fourcc) = formats(descriptor.pixel_format)
        .ok_or_else(|| format!("pixel format {} has no dma-buf format", descriptor.pixel_format))?;
    if descriptor.texture_type != 2 || descriptor.mipmap_level_count != 1 || descriptor.sample_count != 1 {
        return Err("a dma-buf texture is 2D, of one level and one sample".into());
    }
    if metal.resource_options.storage_mode().map_err(|e| e.to_string())? != StorageMode::Private {
        return Err("a dma-buf texture has private storage".into());
    }
    let raw = device.raw();
    if device.device_proc_addr(c"vkGetMemoryFdKHR").is_none()
        || device.device_proc_addr(c"vkGetImageDrmFormatModifierPropertiesEXT").is_none()
    {
        return Err("the device doesn't export dma-bufs with DRM format modifiers".into());
    }
    let vulkan = raw.device.clone();
    let memory_fd = ash::khr::external_memory_fd::Device::new(raw.instance, raw.device);
    let modifiers_fn = ash::ext::image_drm_format_modifier::Device::new(raw.instance, raw.device);

    // MTLTextureUsageShaderWrite is 2.
    let mut usage = vk::ImageUsageFlags::COLOR_ATTACHMENT
        | vk::ImageUsageFlags::SAMPLED
        | vk::ImageUsageFlags::TRANSFER_SRC
        | vk::ImageUsageFlags::TRANSFER_DST;
    if descriptor.usage & 2 != 0 {
        usage |= vk::ImageUsageFlags::STORAGE;
    }
    let linear = [0u64]; // DRM_FORMAT_MOD_LINEAR
    let mut modifiers = vk::ImageDrmFormatModifierListCreateInfoEXT::default().drm_format_modifiers(&linear);
    let mut external =
        vk::ExternalMemoryImageCreateInfo::default().handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
    let families = raw.queue_families;
    let mut info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(format)
        .extent(vk::Extent3D { width: descriptor.width as u32, height: descriptor.height as u32, depth: 1 })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT)
        .usage(usage)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .push_next(&mut external)
        .push_next(&mut modifiers);
    info = if families.len() > 1 {
        info.sharing_mode(vk::SharingMode::CONCURRENT).queue_family_indices(families)
    } else {
        info.sharing_mode(vk::SharingMode::EXCLUSIVE)
    };
    let image = vulkan.create_image(&info, None).map_err(|e| format!("vkCreateImage: {e}"))?;

    // From here, `owned` frees what was made should a later step fail.
    let requirements = vulkan.get_image_memory_requirements(image);
    let properties = raw.instance.get_physical_device_memory_properties(raw.physical_device);
    let types = &properties.memory_types[..properties.memory_type_count as usize];
    let allowed = |i: usize| requirements.memory_type_bits & (1 << i) != 0;
    let Some(type_index) = (0..types.len())
        .find(|&i| allowed(i) && types[i].property_flags.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL))
        .or_else(|| (0..types.len()).find(|&i| allowed(i)))
    else {
        vulkan.destroy_image(image, None);
        return Err("no memory type takes the image".into());
    };
    let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().image(image);
    let mut export =
        vk::ExportMemoryAllocateInfo::default().handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
    let allocation = vk::MemoryAllocateInfo::default()
        .allocation_size(requirements.size)
        .memory_type_index(type_index as u32)
        .push_next(&mut dedicated)
        .push_next(&mut export);
    let memory = match vulkan.allocate_memory(&allocation, None) {
        Ok(memory) => memory,
        Err(e) => {
            vulkan.destroy_image(image, None);
            return Err(format!("vkAllocateMemory: {e}"));
        }
    };
    let owned = DmaBufImage { vulkan: vulkan.clone(), image, memory, _device: device.clone() };
    vulkan.bind_image_memory(image, memory, 0).map_err(|e| format!("vkBindImageMemory: {e}"))?;

    let mut modifier = vk::ImageDrmFormatModifierPropertiesEXT::default();
    modifiers_fn
        .get_image_drm_format_modifier_properties(image, &mut modifier)
        .map_err(|e| format!("vkGetImageDrmFormatModifierPropertiesEXT: {e}"))?;
    let plane = vulkan.get_image_subresource_layout(
        image,
        vk::ImageSubresource { aspect_mask: vk::ImageAspectFlags::MEMORY_PLANE_0_EXT, mip_level: 0, array_layer: 0 },
    );
    let texture = device
        .new_texture_with_vk_image(
            &metal,
            image,
            &info,
            VkImageLayouts::new(metal2vulkan::vk::ImageLayout::UNDEFINED, metal2vulkan::vk::ImageLayout::GENERAL),
        )
        .map_err(|e| e.to_string())?;
    let fd = memory_fd
        .get_memory_fd(
            &vk::MemoryGetFdInfoKHR::default()
                .memory(memory)
                .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT),
        )
        .map_err(|e| format!("vkGetMemoryFdKHR: {e}"))?;
    *dma_buf = [
        fd as i64,
        descriptor.width as i64,
        descriptor.height as i64,
        plane.row_pitch as i64,
        plane.offset as i64,
        fourcc as i64,
        modifier.drm_format_modifier as i64,
    ];
    Ok(M2vTexture(texture, Some(Box::new(owned))))
}

/// `m2v_device_new_dma_buf_texture`: see `dma_buf_texture`. `dma_buf` is seven 64-bit values.
#[no_mangle]
pub unsafe extern "C" fn m2v_device_new_dma_buf_texture(
    device: *const M2vDevice,
    descriptor: *const M2vTextureDescriptor,
    dma_buf: *mut [i64; 7],
    error: *mut *mut c_char,
) -> *mut M2vTexture {
    let device = handle!(device, error, ptr::null_mut());
    let descriptor = handle!(descriptor, error, ptr::null_mut());
    let Some(dma_buf) = dma_buf.as_mut() else {
        return fail(error, "dma_buf is NULL", ptr::null_mut());
    };
    boxed(check!(dma_buf_texture(&device.device, descriptor, dma_buf), error, ptr::null_mut()))
}
