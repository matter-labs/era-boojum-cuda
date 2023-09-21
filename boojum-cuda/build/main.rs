#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

mod gates;
mod poseidon_constants;
mod template;

fn main() {
    gates::generate();
    poseidon_constants::generate();
    #[cfg(target_os = "macos")]
    std::process::exit(0);
    let dst = cmake::Config::new("native")
        .profile("Release")
        .define(
            "CMAKE_CUDA_ARCHITECTURES",
            std::env::var("CUDAARCHS").unwrap_or("native".to_string()),
        )
        .build();
    println!("cargo:rustc-link-search=native={}", dst.display());
    println!("cargo:rustc-link-lib=static=boojum-cuda-native");
    #[cfg(target_os = "windows")]
    println!(
        "cargo:rustc-link-search=native={}",
        concat!(env!("CUDA_PATH"), "/lib/x64")
    );
    #[cfg(target_os = "linux")]
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib");
    println!("cargo:rustc-link-lib=cudart");
    #[cfg(target_os = "linux")]
    println!("cargo:rustc-link-lib=stdc++");
}
