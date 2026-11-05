// The library's soname, as programs that open it name it.
fn main() {
    println!("cargo:rustc-cdylib-link-arg=-Wl,-soname,libmetal2vulkan.so");
}
