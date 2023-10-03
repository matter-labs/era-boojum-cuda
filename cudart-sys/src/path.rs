#[cfg(target_os = "windows")]
#[macro_export]
macro_rules! cuda_path {
    () => {
        env!("CUDA_PATH")
    };
}

#[cfg(target_os = "linux")]
#[macro_export]
macro_rules! cuda_path {
    () => {
        "/usr/local/cuda"
    };
}

#[cfg(not(any(target_os = "windows", target_os = "linux")))]
#[macro_export]
macro_rules! cuda_path {
    () => {
        unimplemented!()
    };
}

#[macro_export]
macro_rules! cuda_include_path {
    () => {
        concat!(cuda_path!(), "/include")
    };
}

#[cfg(target_os = "windows")]
#[macro_export]
macro_rules! cuda_lib_path {
    () => {
        concat!(cuda_path!(), "/lib/x64")
    };
}

#[cfg(target_os = "linux")]
#[macro_export]
macro_rules! cuda_lib_path {
    () => {
        concat!(cuda_path!(), "/lib64")
    };
}

#[cfg(not(any(target_os = "windows", target_os = "linux")))]
#[macro_export]
macro_rules! cuda_lib_path {
    () => {
        unimplemented!()
    };
}
