use std::fs;
use std::path::PathBuf;

use bindgen::callbacks::{EnumVariantValue, ParseCallbacks};

fn cuda_include_path() -> &'static str {
    #[cfg(target_os = "windows")]
    {
        concat!(env!("CUDA_PATH"), "/include")
    }

    #[cfg(target_os = "linux")]
    {
        "/usr/local/cuda/include"
    }

    #[cfg(not(any(target_os = "windows", target_os = "linux")))]
    {
        unimplemented!()
    }
}

fn cuda_lib_path() -> &'static str {
    #[cfg(target_os = "windows")]
    {
        concat!(env!("CUDA_PATH"), "/lib/x64")
    }

    #[cfg(target_os = "linux")]
    {
        "/usr/local/cuda/lib64"
    }

    #[cfg(not(any(target_os = "windows", target_os = "linux")))]
    {
        unimplemented!()
    }
}

#[derive(Debug)]
struct CudaParseCallbacks;

impl ParseCallbacks for CudaParseCallbacks {
    fn enum_variant_name(
        &self,
        enum_name: Option<&str>,
        original_variant_name: &str,
        _variant_value: EnumVariantValue,
    ) -> Option<String> {
        let strip_prefix = |prefix| {
            Some(
                original_variant_name
                    .strip_prefix(prefix)
                    .unwrap()
                    .to_string(),
            )
        };
        if let Some(enum_name) = enum_name {
            match enum_name {
                "enum cudaDeviceAttr" => strip_prefix("cudaDevAttr"),
                "enum cudaLimit" => strip_prefix("cudaLimit"),
                "enum cudaError" => strip_prefix("cuda"),
                "enum cudaMemcpyKind" => strip_prefix("cudaMemcpy"),
                "enum cudaMemPoolAttr" => strip_prefix("cudaMemPool"),
                "enum cudaMemLocationType" => strip_prefix("cudaMemLocationType"),
                "enum cudaMemAllocationType" => strip_prefix("cudaMemAllocationType"),
                "enum cudaMemAllocationHandleType" => strip_prefix("cudaMemHandleType"),
                "enum cudaMemoryType" => strip_prefix("cudaMemoryType"),
                "enum cudaMemAccessFlags" => strip_prefix("cudaMemAccessFlagsProt"),
                "enum cudaFuncAttribute" => strip_prefix("cudaFuncAttribute"),
                "enum cudaFuncCache" => strip_prefix("cudaFuncCache"),
                "enum cudaSharedMemConfig" => strip_prefix("cudaSharedMem"),
                _ => None,
            }
        } else {
            None
        }
    }

    fn item_name(&self, _original_item_name: &str) -> Option<String> {
        let from = |s: &str| Some(String::from(s));
        match _original_item_name {
            "cudaDeviceAttr" => from("CudaDeviceAttr"),
            "cudaLimit" => from("CudaLimit"),
            "cudaError" => from("CudaError"),
            "cudaDeviceProp" => from("CudaDeviceProperties"),
            "cudaMemcpyKind" => from("CudaMemoryCopyKind"),
            "cudaMemPoolProps" => from("CudaMemPoolProperties"),
            "cudaMemPoolAttr" => from("CudaMemPoolAttribute"),
            "cudaMemLocation" => from("CudaMemLocation"),
            "cudaMemLocationType" => from("CudaMemLocationType"),
            "cudaMemAllocationType" => from("CudaMemAllocationType"),
            "cudaMemAllocationHandleType" => from("CudaMemAllocationHandleType"),
            "cudaPointerAttributes" => from("CudaPointerAttributes"),
            "cudaMemoryType" => from("CudaMemoryType"),
            "cudaMemAccessFlags" => from("CudaMemAccessFlags"),
            "cudaMemAccessDesc" => from("CudaMemAccessDesc"),
            "cudaFuncAttributes" => from("CudaFuncAttributes"),
            "cudaFuncAttribute" => from("CudaFuncAttribute"),
            "cudaFuncCache" => from("CudaFuncCache"),
            "cudaSharedMemConfig" => from("CudaSharedMemConfig"),
            _ => None,
        }
    }
}

fn main() {
    #[cfg(target_os = "macos")]
    std::process::exit(0);
    let cuda_lib_path = cuda_lib_path();
    let cuda_runtime_api_path = PathBuf::from(cuda_include_path())
        .join("cuda_runtime_api.h")
        .to_string_lossy()
        .to_string();
    println!("cargo:rustc-link-search=native={cuda_lib_path}");
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rerun-if-changed={cuda_runtime_api_path}");

    let bindings = bindgen::Builder::default()
        .header(cuda_runtime_api_path)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .parse_callbacks(Box::new(CudaParseCallbacks))
        .size_t_is_usize(true)
        .generate_comments(false)
        .layout_tests(false)
        .allowlist_type("cudaError")
        .rustified_enum("cudaError")
        .must_use_type("cudaError")
        // device management
        // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html
        .rustified_enum("cudaDeviceAttr")
        .allowlist_function("cudaDeviceGetAttribute")
        .allowlist_function("cudaDeviceGetDefaultMemPool")
        .rustified_enum("cudaLimit")
        .allowlist_function("cudaDeviceGetLimit")
        .allowlist_function("cudaDeviceGetMemPool")
        .allowlist_function("cudaDeviceReset")
        .allowlist_function("cudaDeviceSetLimit")
        .allowlist_function("cudaDeviceSetMemPool")
        .allowlist_function("cudaDeviceSynchronize")
        .allowlist_function("cudaGetDevice")
        .allowlist_function("cudaGetDeviceCount")
        .allowlist_function("cudaGetDeviceProperties_v2")
        .allowlist_function("cudaSetDevice")
        // error handling
        // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__ERROR.html
        .allowlist_function("cudaGetErrorName")
        .allowlist_function("cudaGetLastError")
        .allowlist_function("cudaPeekAtLastError")
        // stream management
        // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html
        .allowlist_function("cudaStreamCreate")
        .allowlist_var("cudaStreamDefault")
        .allowlist_var("cudaStreamNonBlocking")
        .allowlist_function("cudaStreamCreateWithFlags")
        .allowlist_function("cudaStreamDestroy")
        .allowlist_function("cudaStreamQuery")
        .allowlist_function("cudaStreamSynchronize")
        .allowlist_var("cudaEventWaitDefault")
        .allowlist_var("cudaEventWaitExternal")
        .allowlist_function("cudaStreamWaitEvent")
        // event management
        // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html
        .allowlist_function("cudaEventCreate")
        .allowlist_var("cudaEventDefault")
        .allowlist_var("cudaEventBlockingSync")
        .allowlist_var("cudaEventDisableTiming")
        .allowlist_var("cudaEventInterprocess")
        .allowlist_function("cudaEventCreateWithFlags")
        .allowlist_function("cudaEventDestroy")
        .allowlist_function("cudaEventElapsedTime")
        .allowlist_function("cudaEventQuery")
        .allowlist_function("cudaEventRecord")
        .allowlist_var("cudaEventRecordDefault")
        .allowlist_var("cudaEventRecordExternal")
        .allowlist_function("cudaEventRecordWithFlags")
        .allowlist_function("cudaEventSynchronize")
        // execution control
        // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXECUTION.html
        .rustified_enum("cudaFuncAttribute")
        .allowlist_function("cudaFuncGetAttributes")
        .allowlist_function("cudaFuncSetAttribute")
        .rustified_enum("cudaFuncCache")
        .allowlist_function("cudaFuncSetCacheConfig")
        .rustified_enum("cudaSharedMemConfig")
        .allowlist_function("cudaFuncSetSharedMemConfig")
        .allowlist_function("cudaLaunchHostFunc")
        .allowlist_function("cudaLaunchKernel")
        // occupancy
        // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__OCCUPANCY.html
        .allowlist_function("cudaOccupancyAvailableDynamicSMemPerBlock")
        .allowlist_function("cudaOccupancyMaxActiveBlocksPerMultiprocessor")
        .allowlist_function("cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags")
        .allowlist_function("cudaOccupancyMaxActiveClusters")
        .allowlist_function("cudaOccupancyMaxPotentialClusterSize")
        // memory management
        // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html
        .rustified_enum("cudaMemcpyKind")
        .allowlist_function("cudaFree")
        .allowlist_function("cudaFreeHost")
        .allowlist_function("cudaGetSymbolAddress")
        .allowlist_function("cudaGetSymbolSize")
        .allowlist_var("cudaHostAllocDefault")
        .allowlist_var("cudaHostAllocPortable")
        .allowlist_var("cudaHostAllocMapped")
        .allowlist_var("cudaHostAllocWriteCombined")
        .allowlist_function("cudaHostAlloc")
        .allowlist_var("cudaHostRegisterDefault")
        .allowlist_var("cudaHostRegisterPortable")
        .allowlist_var("cudaHostRegisterMapped")
        .allowlist_var("cudaHostRegisterIoMemory")
        .allowlist_var("cudaHostRegisterReadOnly")
        .allowlist_function("cudaHostRegister")
        .allowlist_function("cudaHostUnregister")
        .allowlist_function("cudaMalloc")
        .allowlist_function("cudaMemGetInfo")
        .allowlist_function("cudaMemcpy")
        .allowlist_function("cudaMemcpyAsync")
        .allowlist_function("cudaMemcpyFromSymbol")
        .allowlist_function("cudaMemcpyFromSymbolAsync")
        .allowlist_function("cudaMemcpyPeer")
        .allowlist_function("cudaMemcpyPeerAsync")
        .allowlist_function("cudaMemcpyToSymbol")
        .allowlist_function("cudaMemcpyToSymbolAsync")
        .allowlist_function("cudaMemset")
        .allowlist_function("cudaMemsetAsync")
        // Stream Ordered Memory Allocator
        // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY__POOLS.html
        .allowlist_function("cudaFreeAsync")
        .allowlist_function("cudaMallocAsync")
        .allowlist_function("cudaMallocFromPoolAsync")
        .rustified_enum("cudaMemLocationType")
        .rustified_enum("cudaMemAllocationType")
        .rustified_enum("cudaMemAllocationHandleType")
        .allowlist_function("cudaMemPoolCreate")
        .allowlist_function("cudaMemPoolDestroy")
        .rustified_enum("cudaMemPoolAttr")
        .rustified_enum("cudaMemAccessFlags")
        .allowlist_function("cudaMemPoolGetAccess")
        .allowlist_function("cudaMemPoolGetAttribute")
        .allowlist_function("cudaMemPoolSetAccess")
        .allowlist_function("cudaMemPoolSetAttribute")
        .allowlist_function("cudaMemPoolTrimTo")
        // Unified Addressing
        // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__UNIFIED.html
        .rustified_enum("cudaMemoryType")
        .allowlist_function("cudaPointerGetAttributes")
        // Peer Device Memory Access
        // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__PEER.html
        .allowlist_function("cudaDeviceCanAccessPeer")
        .allowlist_function("cudaDeviceDisablePeerAccess")
        .allowlist_function("cudaDeviceEnablePeerAccess")
        //
        .generate()
        .expect("Unable to generate bindings");

    fs::write(
        PathBuf::from("src").join("bindings.rs"),
        bindings.to_string(),
    )
    .expect("Couldn't write bindings!");
}
