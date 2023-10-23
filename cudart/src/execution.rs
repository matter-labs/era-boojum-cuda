// execution control
// https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXECUTION.html

use core::ffi::c_void;
use std::marker::PhantomData;
use std::os::raw::{c_char, c_int, c_uint};
use std::sync::{Arc, Weak};

use cudart_sys::*;

use crate::result::{CudaResult, CudaResultWrap};
use crate::stream::CudaStream;

pub struct KernelArguments<'a> {
    vec: Vec<*mut c_void>,
    phantom: PhantomData<&'a c_void>,
}

impl<'a> KernelArguments<'a> {
    pub fn new() -> Self {
        Self {
            vec: vec![],
            phantom: PhantomData,
        }
    }

    pub fn push<T>(&mut self, value: &T) {
        self.vec.push(value as *const T as *mut c_void);
    }

    pub fn as_mut_ptr(&mut self) -> *mut *mut c_void {
        self.vec.as_mut_ptr()
    }
}

impl<'a> Default for KernelArguments<'a> {
    fn default() -> Self {
        KernelArguments::new()
    }
}

#[macro_export]
macro_rules! kernel_args {
    ($($x:expr),* $(,)?) => {
        {
            let mut args = $crate::execution::KernelArguments::new();
            $(
            args.push($x);
            )*
            args
        }
    };
}

pub use kernel_args;

#[derive(Debug, Copy, Clone)]
pub enum CudaLaunchAttribute {
    Ignore,
    AccessPolicyWindow(CudaAccessPolicyWindow),
    Cooperative(bool),
    SynchronizationPolicy(CudaSynchronizationPolicy),
    ClusterDimension(dim3),
    ClusterSchedulingPolicyPreference(CudaClusterSchedulingPolicy),
    ProgrammaticStreamSerialization(bool),
    ProgrammaticEvent(cudaLaunchAttributeValue__bindgen_ty_2),
    Priority(i32),
    MemSyncDomainMap(cudaLaunchMemSyncDomainMap),
    MemSyncDomain(CudaLaunchMemSyncDomain),
}

impl CudaLaunchAttribute {
    pub(crate) fn from_id_and_value(
        id: CudaLaunchAttributeID,
        value: CudaLaunchAttributeValue,
    ) -> Self {
        unsafe {
            match id {
                CudaLaunchAttributeID::Ignore => Self::Ignore,
                CudaLaunchAttributeID::AccessPolicyWindow => {
                    Self::AccessPolicyWindow(value.accessPolicyWindow)
                }
                CudaLaunchAttributeID::Cooperative => Self::Cooperative(value.cooperative != 0),
                CudaLaunchAttributeID::SynchronizationPolicy => {
                    Self::SynchronizationPolicy(value.syncPolicy)
                }
                CudaLaunchAttributeID::ClusterDimension => {
                    let d = value.clusterDim;
                    Self::ClusterDimension(dim3 {
                        x: d.x,
                        y: d.y,
                        z: d.z,
                    })
                }
                CudaLaunchAttributeID::ClusterSchedulingPolicyPreference => {
                    Self::ClusterSchedulingPolicyPreference(value.clusterSchedulingPolicyPreference)
                }
                CudaLaunchAttributeID::ProgrammaticStreamSerialization => {
                    Self::ProgrammaticStreamSerialization(
                        value.programmaticStreamSerializationAllowed != 0,
                    )
                }
                CudaLaunchAttributeID::ProgrammaticEvent => {
                    Self::ProgrammaticEvent(value.programmaticEvent)
                }
                CudaLaunchAttributeID::Priority => Self::Priority(value.priority),
                CudaLaunchAttributeID::MemSyncDomainMap => {
                    Self::MemSyncDomainMap(value.memSyncDomainMap)
                }
                CudaLaunchAttributeID::MemSyncDomain => Self::MemSyncDomain(value.memSyncDomain),
                #[allow(unreachable_patterns)]
                _ => unimplemented!("Unsupported CudaLaunchAttributeID"),
            }
        }
    }

    pub(crate) fn into_id_and_value(self) -> (CudaLaunchAttributeID, CudaLaunchAttributeValue) {
        match self {
            CudaLaunchAttribute::Ignore => (
                CudaLaunchAttributeID::Ignore,
                CudaLaunchAttributeValue { pad: [0; 64] },
            ),
            CudaLaunchAttribute::AccessPolicyWindow(access_policy_window) => (
                CudaLaunchAttributeID::AccessPolicyWindow,
                CudaLaunchAttributeValue {
                    accessPolicyWindow: access_policy_window,
                },
            ),
            CudaLaunchAttribute::Cooperative(cooperative) => (
                CudaLaunchAttributeID::Cooperative,
                CudaLaunchAttributeValue {
                    cooperative: cooperative as c_int,
                },
            ),
            CudaLaunchAttribute::SynchronizationPolicy(sync_policy) => (
                CudaLaunchAttributeID::SynchronizationPolicy,
                CudaLaunchAttributeValue {
                    syncPolicy: sync_policy,
                },
            ),
            CudaLaunchAttribute::ClusterDimension(cluster_dim) => (
                CudaLaunchAttributeID::ClusterDimension,
                CudaLaunchAttributeValue {
                    clusterDim: cudaLaunchAttributeValue__bindgen_ty_1 {
                        x: cluster_dim.x,
                        y: cluster_dim.y,
                        z: cluster_dim.z,
                    },
                },
            ),
            CudaLaunchAttribute::ClusterSchedulingPolicyPreference(
                cluster_scheduling_policy_preference,
            ) => (
                CudaLaunchAttributeID::ClusterSchedulingPolicyPreference,
                CudaLaunchAttributeValue {
                    clusterSchedulingPolicyPreference: cluster_scheduling_policy_preference,
                },
            ),
            CudaLaunchAttribute::ProgrammaticStreamSerialization(
                programmatic_stream_serialization_allowed,
            ) => (
                CudaLaunchAttributeID::ProgrammaticStreamSerialization,
                CudaLaunchAttributeValue {
                    programmaticStreamSerializationAllowed:
                        programmatic_stream_serialization_allowed as c_int,
                },
            ),
            CudaLaunchAttribute::ProgrammaticEvent(programmatic_event) => (
                CudaLaunchAttributeID::ProgrammaticEvent,
                CudaLaunchAttributeValue {
                    programmaticEvent: programmatic_event,
                },
            ),
            CudaLaunchAttribute::Priority(priority) => (
                CudaLaunchAttributeID::Priority,
                CudaLaunchAttributeValue { priority },
            ),
            CudaLaunchAttribute::MemSyncDomainMap(mem_sync_domain_map) => (
                CudaLaunchAttributeID::MemSyncDomainMap,
                CudaLaunchAttributeValue {
                    memSyncDomainMap: mem_sync_domain_map,
                },
            ),
            CudaLaunchAttribute::MemSyncDomain(mem_sync_domain) => (
                CudaLaunchAttributeID::MemSyncDomain,
                CudaLaunchAttributeValue {
                    memSyncDomain: mem_sync_domain,
                },
            ),
        }
    }

    fn into_raw(self) -> cudaLaunchAttribute {
        let (id, val) = self.into_id_and_value();
        cudaLaunchAttribute {
            id,
            pad: [c_char::default(); 4],
            val,
        }
    }
}

pub trait Kernel: Sized {
    fn get_kernel_raw(self) -> *const c_void;
}

pub trait KernelLaunch<'a>: Kernel {
    type Args: Into<KernelArguments<'a>>;

    #[allow(clippy::missing_safety_doc)]
    unsafe fn launch(
        self,
        grid_dim: dim3,
        block_dim: dim3,
        args: Self::Args,
        shared_mem: usize,
        stream: &CudaStream,
    ) -> CudaResult<()> {
        cudaLaunchKernel(
            self.get_kernel_raw(),
            grid_dim,
            block_dim,
            args.into().as_mut_ptr(),
            shared_mem,
            stream.into(),
        )
        .wrap()
    }

    #[allow(clippy::missing_safety_doc)]
    unsafe fn launch_ex(
        self,
        grid_dim: dim3,
        block_dim: dim3,
        args: Self::Args,
        shared_mem: usize,
        stream: &CudaStream,
        attributes: &[CudaLaunchAttribute],
    ) -> CudaResult<()> {
        let mut attributes = attributes
            .iter()
            .map(|attribute| (*attribute).into_raw())
            .collect::<Vec<_>>();
        let config = cudaLaunchConfig_t {
            gridDim: grid_dim,
            blockDim: block_dim,
            dynamicSmemBytes: shared_mem,
            stream: stream.into(),
            attrs: attributes.as_mut_ptr(),
            numAttrs: attributes.len() as c_uint,
        };
        cudaLaunchKernelExC(
            &config as *const _,
            self.get_kernel_raw(),
            args.into().as_mut_ptr(),
        )
        .wrap()
    }
}

impl<'a> From<()> for KernelArguments<'a> {
    fn from(_value: ()) -> Self {
        KernelArguments::default()
    }
}

pub type KernelNoArgs = unsafe extern "C" fn();

impl Kernel for KernelNoArgs {
    fn get_kernel_raw(self) -> *const c_void {
        self as *const c_void
    }
}

impl<'a> KernelLaunch<'a> for KernelNoArgs {
    type Args = ();
}

impl<'a, T> From<(&T,)> for KernelArguments<'a> {
    fn from(value: (&T,)) -> Self {
        kernel_args![value.0]
    }
}

pub type KernelOneArg<T0> = unsafe extern "C" fn(T0);

impl<T0> Kernel for KernelOneArg<T0> {
    fn get_kernel_raw(self) -> *const c_void {
        self as *const c_void
    }
}

impl<'a, T0: 'a> KernelLaunch<'a> for KernelOneArg<T0> {
    type Args = (&'a T0,);
}

impl<'a, T0, T1> From<(&T0, &T1)> for KernelArguments<'a> {
    fn from(value: (&T0, &T1)) -> Self {
        kernel_args![value.0, value.1]
    }
}

pub type KernelTwoArgs<T0, T1> = unsafe extern "C" fn(T0, T1);

impl<'a, T0: 'a, T1: 'a> Kernel for KernelTwoArgs<T0, T1> {
    fn get_kernel_raw(self) -> *const c_void {
        self as *const c_void
    }
}

impl<'a, T0: 'a, T1: 'a> KernelLaunch<'a> for KernelTwoArgs<T0, T1> {
    type Args = (&'a T0, &'a T1);
}

impl<'a, T0, T1, T2> From<(&T0, &T1, &T2)> for KernelArguments<'a> {
    fn from(value: (&T0, &T1, &T2)) -> Self {
        kernel_args![value.0, value.1, value.2]
    }
}

pub type KernelThreeArgs<T0, T1, T2> = unsafe extern "C" fn(T0, T1, T2);

impl<'a, T0: 'a, T1: 'a, T2: 'a> Kernel for KernelThreeArgs<T0, T1, T2> {
    fn get_kernel_raw(self) -> *const c_void {
        self as *const c_void
    }
}

impl<'a, T0: 'a, T1: 'a, T2: 'a> KernelLaunch<'a> for KernelThreeArgs<T0, T1, T2> {
    type Args = (&'a T0, &'a T1, &'a T2);
}

impl<'a, T0, T1, T2, T3> From<(&T0, &T1, &T2, &T3)> for KernelArguments<'a> {
    fn from(value: (&T0, &T1, &T2, &T3)) -> Self {
        kernel_args![value.0, value.1, value.2, value.3]
    }
}

pub type KernelFourArgs<T0, T1, T2, T3> = unsafe extern "C" fn(T0, T1, T2, T3);

impl<'a, T0: 'a, T1: 'a, T2: 'a, T3: 'a> Kernel for KernelFourArgs<T0, T1, T2, T3> {
    fn get_kernel_raw(self) -> *const c_void {
        self as *const c_void
    }
}

impl<'a, T0: 'a, T1: 'a, T2: 'a, T3: 'a> KernelLaunch<'a> for KernelFourArgs<T0, T1, T2, T3> {
    type Args = (&'a T0, &'a T1, &'a T2, &'a T3);
}

impl<'a, T0, T1, T2, T3, T4> From<(&T0, &T1, &T2, &T3, &T4)> for KernelArguments<'a> {
    fn from(value: (&T0, &T1, &T2, &T3, &T4)) -> Self {
        kernel_args![value.0, value.1, value.2, value.3, value.4]
    }
}

pub type KernelFiveArgs<T0, T1, T2, T3, T4> = unsafe extern "C" fn(T0, T1, T2, T3, T4);

impl<'a, T0: 'a, T1: 'a, T2: 'a, T3: 'a, T4: 'a> Kernel for KernelFiveArgs<T0, T1, T2, T3, T4> {
    fn get_kernel_raw(self) -> *const c_void {
        self as *const c_void
    }
}

impl<'a, T0: 'a, T1: 'a, T2: 'a, T3: 'a, T4: 'a> KernelLaunch<'a>
    for KernelFiveArgs<T0, T1, T2, T3, T4>
{
    type Args = (&'a T0, &'a T1, &'a T2, &'a T3, &'a T4);
}

impl<'a, T0, T1, T2, T3, T4, T5> From<(&T0, &T1, &T2, &T3, &T4, &T5)> for KernelArguments<'a> {
    fn from(value: (&T0, &T1, &T2, &T3, &T4, &T5)) -> Self {
        kernel_args![value.0, value.1, value.2, value.3, value.4, value.5]
    }
}

pub type KernelSixArgs<T0, T1, T2, T3, T4, T5> = unsafe extern "C" fn(T0, T1, T2, T3, T4, T5);

impl<'a, T0: 'a, T1: 'a, T2: 'a, T3: 'a, T4: 'a, T5: 'a> Kernel
    for KernelSixArgs<T0, T1, T2, T3, T4, T5>
{
    fn get_kernel_raw(self) -> *const c_void {
        self as *const c_void
    }
}

impl<'a, T0: 'a, T1: 'a, T2: 'a, T3: 'a, T4: 'a, T5: 'a> KernelLaunch<'a>
    for KernelSixArgs<T0, T1, T2, T3, T4, T5>
{
    type Args = (&'a T0, &'a T1, &'a T2, &'a T3, &'a T4, &'a T5);
}

impl<'a, T0, T1, T2, T3, T4, T5, T6> From<(&T0, &T1, &T2, &T3, &T4, &T5, &T6)>
    for KernelArguments<'a>
{
    fn from(value: (&T0, &T1, &T2, &T3, &T4, &T5, &T6)) -> Self {
        kernel_args![value.0, value.1, value.2, value.3, value.4, value.5, value.6]
    }
}

pub type KernelSevenArgs<T0, T1, T2, T3, T4, T5, T6> =
    unsafe extern "C" fn(T0, T1, T2, T3, T4, T5, T6);

impl<'a, T0: 'a, T1: 'a, T2: 'a, T3: 'a, T4: 'a, T5: 'a, T6: 'a> Kernel
    for KernelSevenArgs<T0, T1, T2, T3, T4, T5, T6>
{
    fn get_kernel_raw(self) -> *const c_void {
        self as *const c_void
    }
}

impl<'a, T0: 'a, T1: 'a, T2: 'a, T3: 'a, T4: 'a, T5: 'a, T6: 'a> KernelLaunch<'a>
    for KernelSevenArgs<T0, T1, T2, T3, T4, T5, T6>
{
    type Args = (&'a T0, &'a T1, &'a T2, &'a T3, &'a T4, &'a T5, &'a T6);
}

impl<'a, T0, T1, T2, T3, T4, T5, T6, T7> From<(&T0, &T1, &T2, &T3, &T4, &T5, &T6, &T7)>
    for KernelArguments<'a>
{
    fn from(value: (&T0, &T1, &T2, &T3, &T4, &T5, &T6, &T7)) -> Self {
        kernel_args![value.0, value.1, value.2, value.3, value.4, value.5, value.6, value.7]
    }
}

pub type KernelEightArgs<T0, T1, T2, T3, T4, T5, T6, T7> =
    unsafe extern "C" fn(T0, T1, T2, T3, T4, T5, T6, T7);

impl<'a, T0: 'a, T1: 'a, T2: 'a, T3: 'a, T4: 'a, T5: 'a, T6: 'a, T7: 'a> Kernel
    for KernelEightArgs<T0, T1, T2, T3, T4, T5, T6, T7>
{
    fn get_kernel_raw(self) -> *const c_void {
        self as *const c_void
    }
}

impl<'a, T0: 'a, T1: 'a, T2: 'a, T3: 'a, T4: 'a, T5: 'a, T6: 'a, T7: 'a> KernelLaunch<'a>
    for KernelEightArgs<T0, T1, T2, T3, T4, T5, T6, T7>
{
    type Args = (
        &'a T0,
        &'a T1,
        &'a T2,
        &'a T3,
        &'a T4,
        &'a T5,
        &'a T6,
        &'a T7,
    );
}

pub struct HostFn<'a> {
    arc: Arc<Box<dyn Fn() + Send + 'a>>,
}

impl<'a> HostFn<'a> {
    pub fn new(func: impl Fn() + Send + Sync + 'a) -> Self {
        Self {
            arc: Arc::new(Box::new(func) as Box<dyn Fn() + Send + Sync>),
        }
    }
}

unsafe extern "C" fn launch_host_fn_callback(data: *mut c_void) {
    let raw = data as *const Box<dyn Fn() + Send>;
    let weak = Weak::from_raw(raw);
    if let Some(func) = weak.upgrade() {
        func();
    }
}

pub fn get_raw_fn_and_data(host_fn: &HostFn) -> (cudaHostFn_t, *mut c_void) {
    let weak = Arc::downgrade(&host_fn.arc);
    let raw = weak.into_raw();
    let data = raw as *mut c_void;
    (Some(launch_host_fn_callback), data)
}

pub fn launch_host_fn(stream: &CudaStream, host_fn: &HostFn) -> CudaResult<()> {
    let (func, data) = get_raw_fn_and_data(host_fn);
    unsafe { cudaLaunchHostFunc(stream.into(), func, data).wrap() }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;
    use std::thread;
    use std::time::Duration;

    use serial_test::serial;

    use super::*;

    #[test]
    #[serial]
    fn host_fn_add_executes_one_time() {
        let stream = CudaStream::create().unwrap();
        let mut a = 0;
        let add = || {
            a += 1;
            thread::sleep(Duration::from_millis(10));
        };
        let add_mutex = Mutex::new(add);
        let add_fn = HostFn::new(move || add_mutex.lock().unwrap()());
        let sleep_fn = HostFn::new(|| thread::sleep(Duration::from_millis(10)));
        launch_host_fn(&stream, &add_fn).unwrap();
        stream.synchronize().unwrap();
        launch_host_fn(&stream, &sleep_fn).unwrap();
        launch_host_fn(&stream, &add_fn).unwrap();
        drop(add_fn);
        stream.synchronize().unwrap();
        assert_eq!(a, 1);
    }
}
