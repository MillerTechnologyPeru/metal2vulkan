// Pipeline reflection: MTLRenderPipelineReflection's arguments of each stage,
// flattened into tables. A buffer argument's struct, a member's struct or
// array, and an array's element struct or array are indices into the
// reflection's tables, or -1 for none; a struct's members are contiguous.

use super::*;
use metal2vulkan::reflection::{ArrayType, Binding, StructType};

/// MTLArgument (and MTLBinding).
#[repr(C)]
pub struct M2vArgument {
    name: *mut c_char,
    /// MTLArgumentType (MTLBindingType).
    binding_type: u32,
    index: u32,
    /// MTLArgumentAccess.
    access: u32,
    active: bool,
    is_argument: bool,
    buffer_alignment: u64,
    buffer_data_size: u64,
    /// MTLDataType.
    buffer_data_type: u32,
    buffer_struct: i64,
    /// MTLTextureType and MTLDataType.
    texture_type: u32,
    texture_data_type: u32,
    is_depth_texture: bool,
    array_length: u32,
    threadgroup_alignment: u64,
    threadgroup_data_size: u64,
}

/// MTLStructType.
#[repr(C)]
pub struct M2vStruct {
    first_member: u64,
    member_count: u64,
}

/// MTLStructMember.
#[repr(C)]
pub struct M2vStructMember {
    name: *mut c_char,
    offset: u64,
    /// MTLDataType.
    data_type: u32,
    argument_index: u32,
    struct_type: i64,
    array_type: i64,
}

/// MTLArrayType.
#[repr(C)]
pub struct M2vArrayType {
    /// MTLDataType.
    element_type: u32,
    array_length: u32,
    stride: u64,
    argument_index_stride: u32,
    element_struct: i64,
    element_array: i64,
}

#[repr(C)]
pub struct M2vReflection {
    vertex: *mut M2vArgument,
    vertex_count: usize,
    fragment: *mut M2vArgument,
    fragment_count: usize,
    structs: *mut M2vStruct,
    struct_count: usize,
    members: *mut M2vStructMember,
    member_count: usize,
    arrays: *mut M2vArrayType,
    array_count: usize,
    /// A compute pipeline's kernel arguments; none for a render pipeline's.
    kernel: *mut M2vArgument,
    kernel_count: usize,
}

impl M2vReflection {
    const EMPTY: M2vReflection = M2vReflection {
        vertex: ptr::null_mut(), vertex_count: 0, fragment: ptr::null_mut(), fragment_count: 0,
        structs: ptr::null_mut(), struct_count: 0, members: ptr::null_mut(), member_count: 0,
        arrays: ptr::null_mut(), array_count: 0, kernel: ptr::null_mut(), kernel_count: 0,
    };
}

#[derive(Default)]
struct Tables {
    structs: Vec<M2vStruct>,
    members: Vec<M2vStructMember>,
    arrays: Vec<M2vArrayType>,
}

impl Tables {
    fn add_struct(&mut self, s: &StructType) -> i64 {
        let index = self.structs.len();
        let first = self.members.len();
        self.structs.push(M2vStruct { first_member: first as u64, member_count: s.members().len() as u64 });
        for m in s.members() {
            self.members.push(M2vStructMember {
                name: c_string(m.name()),
                offset: m.offset(),
                data_type: m.data_type().as_raw() as u32,
                argument_index: m.argument_index(),
                struct_type: -1,
                array_type: -1,
            });
        }
        for (i, m) in s.members().iter().enumerate() {
            if let Some(nested) = m.struct_type() {
                let nested = self.add_struct(nested);
                self.members[first + i].struct_type = nested;
            }
            if let Some(array) = m.array_type() {
                let array = self.add_array(array);
                self.members[first + i].array_type = array;
            }
        }
        index as i64
    }

    fn add_array(&mut self, a: &ArrayType) -> i64 {
        let index = self.arrays.len();
        self.arrays.push(M2vArrayType {
            element_type: a.element_type().as_raw() as u32,
            array_length: a.array_length(),
            stride: a.stride(),
            argument_index_stride: a.argument_index_stride(),
            element_struct: -1,
            element_array: -1,
        });
        if let Some(s) = a.element_struct_type() {
            let s = self.add_struct(s);
            self.arrays[index].element_struct = s;
        }
        if let Some(e) = a.element_array_type() {
            let e = self.add_array(e);
            self.arrays[index].element_array = e;
        }
        index as i64
    }

    fn argument(&mut self, b: &Binding) -> M2vArgument {
        let mut out = M2vArgument {
            name: c_string(b.name()),
            binding_type: b.binding_type().as_raw() as u32,
            index: b.index(),
            access: b.access().as_raw() as u32,
            active: b.is_used(),
            is_argument: b.is_argument(),
            buffer_alignment: 0,
            buffer_data_size: 0,
            buffer_data_type: 0,
            buffer_struct: -1,
            texture_type: 0,
            texture_data_type: 0,
            is_depth_texture: false,
            array_length: 1,
            threadgroup_alignment: 0,
            threadgroup_data_size: 0,
        };
        if let Some(buffer) = b.buffer() {
            out.buffer_alignment = buffer.buffer_alignment();
            out.buffer_data_size = buffer.buffer_data_size();
            out.buffer_data_type = buffer.buffer_data_type().as_raw() as u32;
            if let Some(s) = buffer.buffer_struct_type() {
                out.buffer_struct = self.add_struct(s);
            }
        }
        if let Some(texture) = b.texture() {
            out.texture_type = texture.texture_type().as_raw() as u32;
            out.texture_data_type = texture.texture_data_type().as_raw() as u32;
            out.is_depth_texture = texture.is_depth_texture();
            out.array_length = texture.array_length();
        }
        if let Some(threadgroup) = b.threadgroup() {
            out.threadgroup_alignment = threadgroup.threadgroup_memory_alignment();
            out.threadgroup_data_size = threadgroup.threadgroup_memory_data_size();
        }
        out
    }
}

fn leak<T>(list: Vec<T>) -> (*mut T, usize) {
    let count = list.len();
    (Box::into_raw(list.into_boxed_slice()).cast(), count)
}

unsafe fn unleak<T>(list: *mut T, count: usize) -> Box<[T]> {
    if list.is_null() {
        return Box::new([]);
    }
    Box::from_raw(ptr::slice_from_raw_parts_mut(list, count))
}

/// The reflection of a render pipeline state, into *out, which
/// m2v_reflection_free frees.
#[no_mangle]
pub unsafe extern "C" fn m2v_render_pipeline_reflection(
    state: *const render::M2vRenderPipeline,
    out: *mut M2vReflection,
    error: *mut *mut c_char,
) -> bool {
    let state = handle!(state, error, false);
    if out.is_null() {
        return fail(error, "no reflection to fill", false);
    }
    let reflection = check!(state.0.compiled_reflection(), error, false);
    let mut tables = Tables::default();
    let vertex: Vec<_> = reflection.vertex_bindings().iter().map(|b| tables.argument(b)).collect();
    let fragment: Vec<_> = reflection.fragment_bindings().iter().map(|b| tables.argument(b)).collect();
    let (vertex, vertex_count) = leak(vertex);
    let (fragment, fragment_count) = leak(fragment);
    let (structs, struct_count) = leak(tables.structs);
    let (members, member_count) = leak(tables.members);
    let (arrays, array_count) = leak(tables.arrays);
    *out = M2vReflection {
        vertex, vertex_count, fragment, fragment_count, structs, struct_count, members, member_count, arrays,
        array_count, ..M2vReflection::EMPTY
    };
    true
}

/// The reflection of a compute pipeline state, its kernel's arguments, into
/// *out, which m2v_reflection_free frees.
#[no_mangle]
pub unsafe extern "C" fn m2v_compute_pipeline_reflection(
    state: *const M2vComputePipeline,
    out: *mut M2vReflection,
    error: *mut *mut c_char,
) -> bool {
    let state = handle!(state, error, false);
    if out.is_null() {
        return fail(error, "no reflection to fill", false);
    }
    let reflection = check!(state.0.compiled_reflection(), error, false);
    let mut tables = Tables::default();
    let kernel: Vec<_> = reflection.bindings().iter().map(|b| tables.argument(b)).collect();
    let (kernel, kernel_count) = leak(kernel);
    let (structs, struct_count) = leak(tables.structs);
    let (members, member_count) = leak(tables.members);
    let (arrays, array_count) = leak(tables.arrays);
    *out = M2vReflection {
        structs, struct_count, members, member_count, arrays, array_count, kernel, kernel_count,
        ..M2vReflection::EMPTY
    };
    true
}

#[no_mangle]
pub unsafe extern "C" fn m2v_reflection_free(reflection: *mut M2vReflection) {
    let Some(r) = reflection.as_mut() else { return };
    for list in [unleak(r.vertex, r.vertex_count), unleak(r.fragment, r.fragment_count), unleak(r.kernel, r.kernel_count)] {
        for a in list.iter() {
            m2v_string_free(a.name);
        }
    }
    for m in unleak(r.members, r.member_count).iter() {
        m2v_string_free(m.name);
    }
    drop(unleak(r.structs, r.struct_count));
    drop(unleak(r.arrays, r.array_count));
    *r = M2vReflection::EMPTY;
}
