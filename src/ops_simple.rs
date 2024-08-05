use crate::device_structures::{
    DeviceMatrixChunkImpl, DeviceMatrixChunkMutImpl, DeviceRepr, MutPtrAndStrideWrappingMatrix,
    PtrAndStrideWrappingMatrix,
};
use crate::extension_field::VectorizedExtensionField;
use crate::utils::{get_grid_block_dims_for_threads_count, WARP_SIZE};
use crate::BaseField;
use era_cudart::execution::{CudaLaunchConfig, Dim3, KernelFunction};
use era_cudart::memory::memory_set_async;
use era_cudart::paste::paste;
use era_cudart::result::CudaResult;
use era_cudart::slice::DeviceSlice;
use era_cudart::stream::CudaStream;
use era_cudart::{cuda_kernel_declaration, cuda_kernel_signature_arguments_and_function};

type BF = BaseField;
type EF = VectorizedExtensionField;

pub fn set_to_zero<T>(result: &mut DeviceSlice<T>, stream: &CudaStream) -> CudaResult<()> {
    memory_set_async(unsafe { result.transmute_mut() }, 0, stream)
}

fn get_launch_dims(rows: u32, cols: u32) -> (Dim3, Dim3) {
    let (mut grid_dim, block_dim) = get_grid_block_dims_for_threads_count(WARP_SIZE * 4, rows);
    grid_dim.y = cols;
    (grid_dim, block_dim)
}

// SET_BY_VAL_KERNEL
cuda_kernel_signature_arguments_and_function!(
    SetByVal<T: DeviceRepr>,
    value: T,
    result: MutPtrAndStrideWrappingMatrix<<T as DeviceRepr>::Type>,
);

macro_rules! set_by_val_kernel {
    ($type:ty) => {
        paste! {
            cuda_kernel_declaration!(
                [<set_by_val_ $type:lower _kernel>](
                    value: $type,
                    result: MutPtrAndStrideWrappingMatrix<<$type as DeviceRepr>::Type>,
                )
            );
        }
    };
}

pub trait SetByVal: DeviceRepr {
    const KERNEL_FUNCTION: SetByValSignature<Self>;
}

pub fn set_by_val<T: SetByVal>(
    value: T,
    result: &mut (impl DeviceMatrixChunkMutImpl<T> + ?Sized),
    stream: &CudaStream,
) -> CudaResult<()> {
    let result = MutPtrAndStrideWrappingMatrix::new(result);
    let (grid_dim, block_dim) = get_launch_dims(result.rows, result.cols);
    let config = CudaLaunchConfig::basic(grid_dim, block_dim, stream);
    let args = SetByValArguments::new(value, result);
    SetByValFunction(T::KERNEL_FUNCTION).launch(&config, &args)
}

macro_rules! set_by_val_impl {
    ($type:ty) => {
        paste! {
            set_by_val_kernel!($type);
            impl SetByVal for $type {
                const KERNEL_FUNCTION: SetByValSignature<Self> = [<set_by_val_ $type:lower _kernel>];
            }
        }
    };
}

set_by_val_impl!(u32);
set_by_val_impl!(u64);
set_by_val_impl!(BF);
set_by_val_impl!(EF);

// SET_BY_REF_KERNEL
cuda_kernel_signature_arguments_and_function!(
    SetByRef<T: DeviceRepr>,
    values: PtrAndStrideWrappingMatrix<<T as DeviceRepr>::Type>,
    result: MutPtrAndStrideWrappingMatrix<<T as DeviceRepr>::Type>,
);

macro_rules! set_by_ref_kernel {
    ($type:ty) => {
        paste! {
            cuda_kernel_declaration!(
                [<set_by_ref_ $type:lower _kernel>](
                    values: PtrAndStrideWrappingMatrix<<$type as DeviceRepr>::Type>,
                    result: MutPtrAndStrideWrappingMatrix<<$type as DeviceRepr>::Type>,
                )
            );
        }
    };
}

pub trait SetByRef: DeviceRepr {
    const KERNEL_FUNCTION: SetByRefSignature<Self>;
}

pub fn set_by_ref<T: SetByRef>(
    values: &(impl DeviceMatrixChunkImpl<T> + ?Sized),
    result: &mut (impl DeviceMatrixChunkMutImpl<T> + ?Sized),
    stream: &CudaStream,
) -> CudaResult<()> {
    let values = PtrAndStrideWrappingMatrix::new(values);
    let result = MutPtrAndStrideWrappingMatrix::new(result);
    let (grid_dim, block_dim) = get_launch_dims(result.rows, result.cols);
    let config = CudaLaunchConfig::basic(grid_dim, block_dim, stream);
    let args = SetByRefArguments::<T>::new(values, result);
    SetByRefFunction::<T>(T::KERNEL_FUNCTION).launch(&config, &args)
}

macro_rules! set_by_ref_impl {
    ($type:ty) => {
        paste! {
            set_by_ref_kernel!($type);
            impl SetByRef for $type {
                const KERNEL_FUNCTION: SetByRefSignature<Self> = [<set_by_ref_ $type:lower _kernel>];
            }
        }
    };
}

set_by_ref_impl!(u32);
set_by_ref_impl!(u64);
set_by_ref_impl!(BF);
set_by_ref_impl!(EF);

// UNARY_KERNEL
cuda_kernel_signature_arguments_and_function!(
    UnaryOp<T: DeviceRepr>,
    values: PtrAndStrideWrappingMatrix<<T as DeviceRepr>::Type>,
    result: MutPtrAndStrideWrappingMatrix<<T as DeviceRepr>::Type>,
);

macro_rules! unary_op_kernel {
    ($op:ty, $type:ty) => {
        paste! {
            cuda_kernel_declaration!(
                [<$op:lower _ $type:lower _kernel>](
                    values: PtrAndStrideWrappingMatrix<<$type as DeviceRepr>::Type>,
                    result: MutPtrAndStrideWrappingMatrix<<$type as DeviceRepr>::Type>,
                )
            );
        }
    };
}

pub trait UnaryOp<T: DeviceRepr> {
    const KERNEL_FUNCTION: UnaryOpSignature<T>;

    fn launch_op(
        values: PtrAndStrideWrappingMatrix<T::Type>,
        result: MutPtrAndStrideWrappingMatrix<T::Type>,
        stream: &CudaStream,
    ) -> CudaResult<()> {
        let (grid_dim, block_dim) = get_launch_dims(result.rows, result.cols);
        let config = CudaLaunchConfig::basic(grid_dim, block_dim, stream);
        let args = UnaryOpArguments::<T>::new(values, result);
        UnaryOpFunction::<T>(Self::KERNEL_FUNCTION).launch(&config, &args)
    }

    fn launch(
        values: &(impl DeviceMatrixChunkImpl<T> + ?Sized),
        result: &mut (impl DeviceMatrixChunkMutImpl<T> + ?Sized),
        stream: &CudaStream,
    ) -> CudaResult<()> {
        assert_eq!(result.rows() % values.rows(), 0);
        assert_eq!(result.cols() % values.cols(), 0);
        Self::launch_op(
            PtrAndStrideWrappingMatrix::new(values),
            MutPtrAndStrideWrappingMatrix::new(result),
            stream,
        )
    }

    fn launch_in_place(
        values: &mut (impl DeviceMatrixChunkMutImpl<T> + ?Sized),
        stream: &CudaStream,
    ) -> CudaResult<()> {
        Self::launch_op(
            PtrAndStrideWrappingMatrix::new(values),
            MutPtrAndStrideWrappingMatrix::new(values),
            stream,
        )
    }
}

macro_rules! unary_op_def {
    ($op:ty) => {
        paste! {
            pub struct $op;
            pub fn [<$op:lower>]<T: DeviceRepr>(
                values: &(impl DeviceMatrixChunkImpl<T> + ?Sized),
                result: &mut (impl DeviceMatrixChunkMutImpl<T> + ?Sized),
                stream: &CudaStream,
            ) -> CudaResult<()> where $op: UnaryOp<T> {
                $op::launch(values, result, stream)
            }
            pub fn [<$op:lower _in_place>]<T: DeviceRepr>(
                values: &mut (impl DeviceMatrixChunkMutImpl<T> + ?Sized),
                stream: &CudaStream,
            ) -> CudaResult<()>  where $op: UnaryOp<T> {
                $op::launch_in_place(values, stream)
            }
        }
    };
}

unary_op_def!(Dbl);
unary_op_def!(Inv);
unary_op_def!(Neg);
unary_op_def!(Sqr);

macro_rules! unary_op_impl {
    ($op:ty, $type:ty) => {
        paste! {
            unary_op_kernel!($op, $type);
            impl UnaryOp<$type> for $op {
                const KERNEL_FUNCTION: UnaryOpSignature<$type> = [<$op:lower _ $type:lower _kernel>];
            }
        }
    };
}

macro_rules! unary_ops_impl {
    ($type:ty) => {
        unary_op_impl!(Dbl, $type);
        unary_op_impl!(Inv, $type);
        unary_op_impl!(Neg, $type);
        unary_op_impl!(Sqr, $type);
    };
}

unary_ops_impl!(BF);
unary_ops_impl!(EF);

// PARAMETRIZED_KERNEL
cuda_kernel_signature_arguments_and_function!(
    ParametrizedOp<T: DeviceRepr>,
    values: PtrAndStrideWrappingMatrix<<T as DeviceRepr>::Type>,
    param: u32,
    result: MutPtrAndStrideWrappingMatrix<<T as DeviceRepr>::Type>,
);

macro_rules! parametrized_op_kernel {
    ($op:ty, $type:ty) => {
        paste! {
            cuda_kernel_declaration!(
                [<$op:lower _ $type:lower _kernel>](
                    values: PtrAndStrideWrappingMatrix<<$type as DeviceRepr>::Type>,
                    param: u32,
                    result: MutPtrAndStrideWrappingMatrix<<$type as DeviceRepr>::Type>,
                )
            );
        }
    };
}

pub trait ParametrizedOp<T: DeviceRepr> {
    const KERNEL_FUNCTION: ParametrizedOpSignature<T>;

    fn launch_op(
        values: PtrAndStrideWrappingMatrix<T::Type>,
        param: u32,
        result: MutPtrAndStrideWrappingMatrix<T::Type>,
        stream: &CudaStream,
    ) -> CudaResult<()> {
        let (grid_dim, block_dim) = get_launch_dims(result.rows, result.cols);
        let config = CudaLaunchConfig::basic(grid_dim, block_dim, stream);
        let args = ParametrizedOpArguments::<T>::new(values, param, result);
        ParametrizedOpFunction::<T>(Self::KERNEL_FUNCTION).launch(&config, &args)
    }

    fn launch(
        values: &(impl DeviceMatrixChunkImpl<T> + ?Sized),
        param: u32,
        result: &mut (impl DeviceMatrixChunkMutImpl<T> + ?Sized),
        stream: &CudaStream,
    ) -> CudaResult<()> {
        assert_eq!(result.rows() % values.rows(), 0);
        assert_eq!(result.cols() % values.cols(), 0);
        Self::launch_op(
            PtrAndStrideWrappingMatrix::new(values),
            param,
            MutPtrAndStrideWrappingMatrix::new(result),
            stream,
        )
    }

    fn launch_in_place(
        values: &mut (impl DeviceMatrixChunkMutImpl<T> + ?Sized),
        param: u32,
        stream: &CudaStream,
    ) -> CudaResult<()> {
        Self::launch_op(
            PtrAndStrideWrappingMatrix::new(values),
            param,
            MutPtrAndStrideWrappingMatrix::new(values),
            stream,
        )
    }
}

macro_rules! parametrized_op_def {
    ($op:ty) => {
        paste! {
            pub struct $op;
            pub fn [<$op:lower>]<T: DeviceRepr>(
                values: &(impl DeviceMatrixChunkImpl<T> + ?Sized),
                param: u32,
                result: &mut (impl DeviceMatrixChunkMutImpl<T> + ?Sized),
                stream: &CudaStream,
            ) -> CudaResult<()>  where $op: ParametrizedOp<T> {
                $op::launch(values, param, result, stream)
            }
            pub fn [<$op:lower _in_place>]<T: DeviceRepr>(
                values: &mut (impl DeviceMatrixChunkMutImpl<T> + ?Sized),
                param: u32,
                stream: &CudaStream,
            ) -> CudaResult<()>  where $op: ParametrizedOp<T> {
                $op::launch_in_place(values, param, stream)
            }
        }
    };
}

parametrized_op_def!(Pow);
parametrized_op_def!(Shl);
parametrized_op_def!(Shr);

macro_rules! parametrized_op_impl {
    ($op:ty, $type:ty) => {
        paste! {
            parametrized_op_kernel!($op, $type);
            impl ParametrizedOp<$type> for $op {
                const KERNEL_FUNCTION: ParametrizedOpSignature<$type> = [<$op:lower _ $type:lower _kernel>];
            }
        }
    };
}

macro_rules! parametrized_ops_impl {
    ($type:ty) => {
        parametrized_op_impl!(Pow, $type);
        parametrized_op_impl!(Shl, $type);
        parametrized_op_impl!(Shr, $type);
    };
}

parametrized_ops_impl!(BF);
parametrized_ops_impl!(EF);

// BINARY_KERNEL
cuda_kernel_signature_arguments_and_function!(
    BinaryOp<T0: DeviceRepr, T1: DeviceRepr, TR: DeviceRepr>,
    x: PtrAndStrideWrappingMatrix<<T0 as DeviceRepr>::Type>,
    y: PtrAndStrideWrappingMatrix<<T1 as DeviceRepr>::Type>,
    result: MutPtrAndStrideWrappingMatrix<<TR as DeviceRepr>::Type>,
);

macro_rules! binary_op_kernel {
    ($op:ty, $t0:ty, $t1:ty, $tr:ty) => {
        paste! {
            cuda_kernel_declaration!(
                [<$op:lower _ $t0:lower _ $t1:lower _kernel>](
                    x: PtrAndStrideWrappingMatrix<<$t0 as DeviceRepr>::Type>,
                    y: PtrAndStrideWrappingMatrix<<$t1 as DeviceRepr>::Type>,
                    result: MutPtrAndStrideWrappingMatrix<<$tr as DeviceRepr>::Type>,
                )
            );
        }
    };
}

pub trait BinaryOp<T0: DeviceRepr, T1: DeviceRepr, TR: DeviceRepr> {
    const KERNEL_FUNCTION: BinaryOpSignature<T0, T1, TR>;

    fn launch_op(
        x: PtrAndStrideWrappingMatrix<T0::Type>,
        y: PtrAndStrideWrappingMatrix<T1::Type>,
        result: MutPtrAndStrideWrappingMatrix<TR::Type>,
        stream: &CudaStream,
    ) -> CudaResult<()> {
        let (grid_dim, block_dim) = get_launch_dims(result.rows, result.cols);
        let config = CudaLaunchConfig::basic(grid_dim, block_dim, stream);
        let args = BinaryOpArguments::<T0, T1, TR>::new(x, y, result);
        BinaryOpFunction::<T0, T1, TR>(Self::KERNEL_FUNCTION).launch(&config, &args)
    }

    fn launch(
        x: &(impl DeviceMatrixChunkImpl<T0> + ?Sized),
        y: &(impl DeviceMatrixChunkImpl<T1> + ?Sized),
        result: &mut (impl DeviceMatrixChunkMutImpl<TR> + ?Sized),
        stream: &CudaStream,
    ) -> CudaResult<()> {
        assert_eq!(result.rows() % x.rows(), 0);
        assert_eq!(result.cols() % x.cols(), 0);
        assert_eq!(result.rows() % y.rows(), 0);
        assert_eq!(result.cols() % y.cols(), 0);
        Self::launch_op(
            PtrAndStrideWrappingMatrix::new(x),
            PtrAndStrideWrappingMatrix::new(y),
            MutPtrAndStrideWrappingMatrix::new(result),
            stream,
        )
    }

    fn launch_into_x(
        x: &mut (impl DeviceMatrixChunkMutImpl<T0> + ?Sized),
        y: &(impl DeviceMatrixChunkImpl<T1> + ?Sized),
        stream: &CudaStream,
    ) -> CudaResult<()>
    where
        Self: BinaryOp<T0, T1, T0>,
    {
        <Self as BinaryOp<T0, T1, T0>>::launch_op(
            PtrAndStrideWrappingMatrix::new(x),
            PtrAndStrideWrappingMatrix::new(y),
            MutPtrAndStrideWrappingMatrix::new(x),
            stream,
        )
    }

    fn launch_into_y(
        x: &(impl DeviceMatrixChunkImpl<T0> + ?Sized),
        y: &mut (impl DeviceMatrixChunkMutImpl<T1> + ?Sized),
        stream: &CudaStream,
    ) -> CudaResult<()>
    where
        Self: BinaryOp<T0, T1, T1>,
    {
        <Self as BinaryOp<T0, T1, T1>>::launch_op(
            PtrAndStrideWrappingMatrix::new(x),
            PtrAndStrideWrappingMatrix::new(y),
            MutPtrAndStrideWrappingMatrix::new(y),
            stream,
        )
    }
}

macro_rules! binary_op_def {
    ($op:ty) => {
        paste! {
            pub struct $op;
            pub fn [<$op:lower>]<T0: DeviceRepr, T1: DeviceRepr, TR: DeviceRepr>(
                x: &(impl DeviceMatrixChunkImpl<T0> + ?Sized),
                y: &(impl DeviceMatrixChunkImpl<T1> + ?Sized),
                result: &mut (impl DeviceMatrixChunkMutImpl<TR> + ?Sized),
                stream: &CudaStream,
            ) -> CudaResult<()> where $op: BinaryOp<T0, T1, TR> {
                $op::launch(x, y, result, stream)
            }
            pub fn [<$op:lower _into_x>]<T0: DeviceRepr, T1: DeviceRepr>(
                x: &mut (impl DeviceMatrixChunkMutImpl<T0> + ?Sized),
                y: &(impl DeviceMatrixChunkImpl<T1> + ?Sized),
                stream: &CudaStream,
            ) -> CudaResult<()>  where $op: BinaryOp<T0, T1, T0> {
                $op::launch_into_x(x, y, stream)
            }
            pub fn [<$op:lower _into_y>]<T0: DeviceRepr, T1: DeviceRepr>(
                x: &(impl DeviceMatrixChunkImpl<T0> + ?Sized),
                y: &mut (impl DeviceMatrixChunkMutImpl<T1> + ?Sized),
                stream: &CudaStream,
            ) -> CudaResult<()>  where $op: BinaryOp<T0, T1, T1> {
                $op::launch_into_y(x, y, stream)
            }
        }
    };
}

binary_op_def!(Add);
binary_op_def!(Mul);
binary_op_def!(Sub);

macro_rules! binary_op_impl {
    ($op:ty, $t0:ty, $t1:ty, $tr:ty) => {
        paste! {
            binary_op_kernel!($op, $t0, $t1, $tr);
            impl BinaryOp<$t0, $t1, $tr> for $op {
                const KERNEL_FUNCTION: BinaryOpSignature<$t0, $t1, $tr> = [<$op:lower _ $t0:lower _ $t1:lower _kernel>];
            }
        }
    };
}

macro_rules! binary_ops_impl {
    ($t0:ty, $t1:ty, $tr:ty) => {
        binary_op_impl!(Add, $t0, $t1, $tr);
        binary_op_impl!(Mul, $t0, $t1, $tr);
        binary_op_impl!(Sub, $t0, $t1, $tr);
    };
}

binary_ops_impl!(BF, BF, BF);
binary_ops_impl!(BF, EF, EF);
binary_ops_impl!(EF, BF, EF);
binary_ops_impl!(EF, EF, EF);

// TERNARY_KERNEL
cuda_kernel_signature_arguments_and_function!(
    TernaryOp<T0: DeviceRepr, T1: DeviceRepr, T2: DeviceRepr, TR: DeviceRepr>,
    x: PtrAndStrideWrappingMatrix<<T0 as DeviceRepr>::Type>,
    y: PtrAndStrideWrappingMatrix<<T1 as DeviceRepr>::Type>,
    z: PtrAndStrideWrappingMatrix<<T2 as DeviceRepr>::Type>,
    result: MutPtrAndStrideWrappingMatrix<<TR as DeviceRepr>::Type>,
);

macro_rules! ternary_op_kernel {
    ($fn_name:ident, $t0:ty, $t1:ty, $t2:ty, $tr:ty) => {
        paste! {
            cuda_kernel_declaration!(
                [<$fn_name _ $t0:lower _ $t1:lower _ $t2:lower _kernel>](
                    x: PtrAndStrideWrappingMatrix<<$t0 as DeviceRepr>::Type>,
                    y: PtrAndStrideWrappingMatrix<<$t1 as DeviceRepr>::Type>,
                    z: PtrAndStrideWrappingMatrix<<$t2 as DeviceRepr>::Type>,
                    result: MutPtrAndStrideWrappingMatrix<<$tr as DeviceRepr>::Type>,
                )
            );
        }
    };
}

pub trait TernaryOp<T0: DeviceRepr, T1: DeviceRepr, T2: DeviceRepr, TR: DeviceRepr> {
    fn get_kernel_function() -> TernaryOpSignature<T0, T1, T2, TR>;

    fn launch_op(
        x: PtrAndStrideWrappingMatrix<T0::Type>,
        y: PtrAndStrideWrappingMatrix<T1::Type>,
        z: PtrAndStrideWrappingMatrix<T2::Type>,
        result: MutPtrAndStrideWrappingMatrix<TR::Type>,
        stream: &CudaStream,
    ) -> CudaResult<()> {
        let kernel_function = Self::get_kernel_function();
        let (grid_dim, block_dim) = get_launch_dims(result.rows, result.cols);
        let config = CudaLaunchConfig::basic(grid_dim, block_dim, stream);
        let args = TernaryOpArguments::<T0, T1, T2, TR>::new(x, y, z, result);
        TernaryOpFunction::<T0, T1, T2, TR>(kernel_function).launch(&config, &args)
    }

    fn launch(
        x: &(impl DeviceMatrixChunkImpl<T0> + ?Sized),
        y: &(impl DeviceMatrixChunkImpl<T1> + ?Sized),
        z: &(impl DeviceMatrixChunkImpl<T2> + ?Sized),
        result: &mut (impl DeviceMatrixChunkMutImpl<TR> + ?Sized),
        stream: &CudaStream,
    ) -> CudaResult<()> {
        assert_eq!(result.rows() % x.rows(), 0);
        assert_eq!(result.cols() % x.cols(), 0);
        assert_eq!(result.rows() % y.rows(), 0);
        assert_eq!(result.cols() % y.cols(), 0);
        assert_eq!(result.rows() % z.rows(), 0);
        assert_eq!(result.cols() % z.cols(), 0);
        Self::launch_op(
            PtrAndStrideWrappingMatrix::new(x),
            PtrAndStrideWrappingMatrix::new(y),
            PtrAndStrideWrappingMatrix::new(z),
            MutPtrAndStrideWrappingMatrix::new(result),
            stream,
        )
    }

    fn launch_into_x(
        x: &mut (impl DeviceMatrixChunkMutImpl<T0> + ?Sized),
        y: &(impl DeviceMatrixChunkImpl<T1> + ?Sized),
        z: &(impl DeviceMatrixChunkImpl<T2> + ?Sized),
        stream: &CudaStream,
    ) -> CudaResult<()>
    where
        Self: TernaryOp<T0, T1, T2, T0>,
    {
        <Self as TernaryOp<T0, T1, T2, T0>>::launch_op(
            PtrAndStrideWrappingMatrix::new(x),
            PtrAndStrideWrappingMatrix::new(y),
            PtrAndStrideWrappingMatrix::new(z),
            MutPtrAndStrideWrappingMatrix::new(x),
            stream,
        )
    }

    fn launch_into_y(
        x: &(impl DeviceMatrixChunkImpl<T0> + ?Sized),
        y: &mut (impl DeviceMatrixChunkMutImpl<T1> + ?Sized),
        z: &(impl DeviceMatrixChunkImpl<T2> + ?Sized),
        stream: &CudaStream,
    ) -> CudaResult<()>
    where
        Self: TernaryOp<T0, T1, T2, T1>,
    {
        <Self as TernaryOp<T0, T1, T2, T1>>::launch_op(
            PtrAndStrideWrappingMatrix::new(x),
            PtrAndStrideWrappingMatrix::new(y),
            PtrAndStrideWrappingMatrix::new(z),
            MutPtrAndStrideWrappingMatrix::new(y),
            stream,
        )
    }

    fn launch_into_z(
        x: &(impl DeviceMatrixChunkImpl<T0> + ?Sized),
        y: &(impl DeviceMatrixChunkImpl<T1> + ?Sized),
        z: &mut (impl DeviceMatrixChunkMutImpl<T2> + ?Sized),
        stream: &CudaStream,
    ) -> CudaResult<()>
    where
        Self: TernaryOp<T0, T1, T2, T2>,
    {
        <Self as TernaryOp<T0, T1, T2, T2>>::launch_op(
            PtrAndStrideWrappingMatrix::new(x),
            PtrAndStrideWrappingMatrix::new(y),
            PtrAndStrideWrappingMatrix::new(z),
            MutPtrAndStrideWrappingMatrix::new(z),
            stream,
        )
    }
}

macro_rules! ternary_op_def {
    ($op:ty, $fn_name:ident) => {
        paste! {
            pub struct $op;
            pub fn $fn_name<T0: DeviceRepr, T1: DeviceRepr, T2: DeviceRepr, TR: DeviceRepr>(
                x: &(impl DeviceMatrixChunkImpl<T0> + ?Sized),
                y: &(impl DeviceMatrixChunkImpl<T1> + ?Sized),
                z: &(impl DeviceMatrixChunkImpl<T2> + ?Sized),
                result: &mut (impl DeviceMatrixChunkMutImpl<TR> + ?Sized),
                stream: &CudaStream,
            ) -> CudaResult<()> where $op: TernaryOp<T0, T1, T2, TR> {
                $op::launch(x, y, z, result, stream)
            }
            pub fn [<$fn_name _into_x>]<T0: DeviceRepr, T1: DeviceRepr, T2: DeviceRepr>(
                x: &mut (impl DeviceMatrixChunkMutImpl<T0> + ?Sized),
                y: &(impl DeviceMatrixChunkImpl<T1> + ?Sized),
                z: &(impl DeviceMatrixChunkImpl<T2> + ?Sized),
                stream: &CudaStream,
            ) -> CudaResult<()>  where $op: TernaryOp<T0, T1, T2, T0> {
                $op::launch_into_x(x, y, z, stream)
            }
            pub fn [<$fn_name _into_y>]<T0: DeviceRepr, T1: DeviceRepr, T2: DeviceRepr>(
                x: &(impl DeviceMatrixChunkImpl<T0> + ?Sized),
                y: &mut (impl DeviceMatrixChunkMutImpl<T1> + ?Sized),
                z: &(impl DeviceMatrixChunkImpl<T2> + ?Sized),
                stream: &CudaStream,
            ) -> CudaResult<()>  where $op: TernaryOp<T0, T1, T2, T1> {
                $op::launch_into_y(x, y, z, stream)
            }
            pub fn [<$fn_name _into_z>]<T0: DeviceRepr, T1: DeviceRepr, T2: DeviceRepr>(
                x: &(impl DeviceMatrixChunkImpl<T0> + ?Sized),
                y: &(impl DeviceMatrixChunkImpl<T1> + ?Sized),
                z: &mut (impl DeviceMatrixChunkMutImpl<T2> + ?Sized),
                stream: &CudaStream,
            ) -> CudaResult<()>  where $op: TernaryOp<T0, T1, T2, T2> {
                $op::launch_into_z(x, y, z, stream)
            }
        }
    };
}

ternary_op_def!(MulAdd, mul_add);
ternary_op_def!(MulSub, mul_sub);

macro_rules! ternary_op_impl {
    ($op:ty, $fn_name:ident, $t0:ty, $t1:ty, $t2:ty, $tr:ty) => {
        paste! {
            ternary_op_kernel!($fn_name, $t0, $t1, $t2, $tr);
            impl TernaryOp<$t0, $t1, $t2, $tr> for $op {
                fn get_kernel_function() -> TernaryOpSignature<$t0, $t1, $t2, $tr> {
                    [<$fn_name _ $t0:lower _ $t1:lower _ $t2:lower _kernel>]
                }
            }
        }
    };
}

macro_rules! ternary_ops_impl {
    ($t0:ty, $t1:ty, $t2:ty, $tr:ty) => {
        ternary_op_impl!(MulAdd, mul_add, $t0, $t1, $t2, $tr);
        ternary_op_impl!(MulSub, mul_sub, $t0, $t1, $t2, $tr);
    };
}

ternary_ops_impl!(BF, BF, BF, BF);
ternary_ops_impl!(BF, BF, EF, EF);
ternary_ops_impl!(BF, EF, BF, EF);
ternary_ops_impl!(BF, EF, EF, EF);
ternary_ops_impl!(EF, BF, BF, EF);
ternary_ops_impl!(EF, BF, EF, EF);
ternary_ops_impl!(EF, EF, BF, EF);
ternary_ops_impl!(EF, EF, EF, EF);

#[cfg(test)]
mod tests {
    use boojum::field::goldilocks::GoldilocksField;
    use boojum::field::{Field, PrimeField, U64Representable};
    use era_cudart::memory::{memory_copy_async, DeviceAllocation};
    use era_cudart::result::CudaResult;
    use era_cudart::slice::DeviceSlice;
    use era_cudart::stream::CudaStream;
    use itertools::Itertools;
    use std::ops::{Add, Mul, Sub};

    #[test]
    fn set_by_val() {
        const N: usize = 1 << 10;
        const VALUE: GoldilocksField = GoldilocksField::ONE;
        let stream = CudaStream::default();
        let mut dst_host = [GoldilocksField::ZERO; N];
        let mut dst_device = DeviceAllocation::<GoldilocksField>::alloc(N).unwrap();
        memory_copy_async(&mut dst_device, &dst_host, &stream).unwrap();
        super::set_by_val(VALUE, &mut dst_device, &stream).unwrap();
        memory_copy_async(&mut dst_host, &dst_device, &stream).unwrap();
        stream.synchronize().unwrap();
        assert!(dst_host.iter().all(|x| { x.eq(&VALUE) }));
    }

    #[test]
    fn set_by_ref() {
        const N: usize = 1 << 10;
        const VALUE: GoldilocksField = GoldilocksField::ONE;
        let stream = CudaStream::default();
        let mut value_device = DeviceAllocation::<GoldilocksField>::alloc(1).unwrap();
        super::set_by_val(VALUE, &mut value_device, &stream).unwrap();
        let mut dst_host = [GoldilocksField::ZERO; N];
        let mut dst_device = DeviceAllocation::<GoldilocksField>::alloc(N).unwrap();
        memory_copy_async(&mut dst_device, &dst_host, &stream).unwrap();
        super::set_by_ref(&value_device, &mut dst_device, &stream).unwrap();
        memory_copy_async(&mut dst_host, &dst_device, &stream).unwrap();
        stream.synchronize().unwrap();
        assert!(dst_host.iter().all(|x| { x.eq(&VALUE) }));
    }

    #[test]
    fn set_to_zero() {
        const N: usize = 1 << 10;
        let stream = CudaStream::default();
        let mut dst_host = [GoldilocksField::ONE; N];
        let mut dst_device = DeviceAllocation::<GoldilocksField>::alloc(N).unwrap();
        memory_copy_async(&mut dst_device, &dst_host, &stream).unwrap();
        super::set_to_zero(&mut dst_device, &stream).unwrap();
        memory_copy_async(&mut dst_host, &dst_device, &stream).unwrap();
        stream.synchronize().unwrap();
        assert!(dst_host.iter().all(|x| { x.eq(&GoldilocksField::ZERO) }));
    }

    type UnaryDeviceFn = fn(
        &DeviceSlice<GoldilocksField>,
        &mut DeviceSlice<GoldilocksField>,
        &CudaStream,
    ) -> CudaResult<()>;

    type UnaryDeviceInPlaceFn =
        fn(&mut DeviceSlice<GoldilocksField>, &CudaStream) -> CudaResult<()>;

    type UnaryHostFn = fn(&GoldilocksField) -> GoldilocksField;

    type ParametrizedUnaryDeviceFn = fn(
        &DeviceSlice<GoldilocksField>,
        u32,
        &mut DeviceSlice<GoldilocksField>,
        &CudaStream,
    ) -> CudaResult<()>;

    type ParametrizedUnaryDeviceInPlaceFn =
        fn(&mut DeviceSlice<GoldilocksField>, u32, &CudaStream) -> CudaResult<()>;

    type ParametrizedUnaryHostFn = fn(&GoldilocksField, u32) -> GoldilocksField;

    type BinaryDeviceFn = fn(
        &DeviceSlice<GoldilocksField>,
        &DeviceSlice<GoldilocksField>,
        &mut DeviceSlice<GoldilocksField>,
        &CudaStream,
    ) -> CudaResult<()>;

    type BinaryDeviceInPlaceFn = fn(
        &mut DeviceSlice<GoldilocksField>,
        &DeviceSlice<GoldilocksField>,
        &CudaStream,
    ) -> CudaResult<()>;

    type BinaryHostFn = fn(&GoldilocksField, &GoldilocksField) -> GoldilocksField;

    type TernaryDeviceFn = fn(
        &DeviceSlice<GoldilocksField>,
        &DeviceSlice<GoldilocksField>,
        &DeviceSlice<GoldilocksField>,
        &mut DeviceSlice<GoldilocksField>,
        &CudaStream,
    ) -> CudaResult<()>;

    type TernaryDeviceInPlaceFn = fn(
        &mut DeviceSlice<GoldilocksField>,
        &DeviceSlice<GoldilocksField>,
        &DeviceSlice<GoldilocksField>,
        &CudaStream,
    ) -> CudaResult<()>;

    type TernaryHostFn =
        fn(&GoldilocksField, &GoldilocksField, &GoldilocksField) -> GoldilocksField;

    fn unary_op_test(
        values: &[u64],
        device_fn: UnaryDeviceFn,
        host_fn: UnaryHostFn,
        compare_reduced: bool,
    ) {
        let x_host: Vec<GoldilocksField> = values.iter().map(|v| GoldilocksField(*v)).collect();
        let stream = CudaStream::default();
        let length = x_host.len();
        let mut result_host = vec![GoldilocksField::ZERO; length];
        let mut x_device = DeviceAllocation::<GoldilocksField>::alloc(length).unwrap();
        let mut result_device = DeviceAllocation::<GoldilocksField>::alloc(length).unwrap();
        memory_copy_async(&mut x_device, &x_host, &stream).unwrap();
        device_fn(&x_device, &mut result_device, &stream).unwrap();
        memory_copy_async(&mut result_host, &result_device, &stream).unwrap();
        stream.synchronize().unwrap();
        for i in 0..length {
            let left = host_fn(&x_host[i]);
            let right = result_host[i];
            if compare_reduced {
                assert_eq!(left, right);
            } else {
                assert_eq!(left.as_u64(), right.as_u64());
            }
        }
    }

    fn unary_op_in_place_test(
        values: &[u64],
        device_fn: UnaryDeviceInPlaceFn,
        host_fn: UnaryHostFn,
        compare_reduced: bool,
    ) {
        let x_host: Vec<GoldilocksField> = values.iter().map(|v| GoldilocksField(*v)).collect();
        let stream = CudaStream::default();
        let length = x_host.len();
        let mut result_host = vec![GoldilocksField::ZERO; length];
        let mut x_device = DeviceAllocation::<GoldilocksField>::alloc(length).unwrap();
        memory_copy_async(&mut x_device, &x_host, &stream).unwrap();
        device_fn(&mut x_device, &stream).unwrap();
        memory_copy_async(&mut result_host, &x_device, &stream).unwrap();
        stream.synchronize().unwrap();
        for i in 0..length {
            let left = host_fn(&x_host[i]);
            let right = result_host[i];
            if compare_reduced {
                assert_eq!(left, right);
            } else {
                assert_eq!(left.as_u64(), right.as_u64());
            }
        }
    }

    fn parametrized_unary_op_test(
        values: &[u64],
        parameter: u32,
        device_fn: ParametrizedUnaryDeviceFn,
        host_fn: ParametrizedUnaryHostFn,
        compare_reduced: bool,
    ) {
        let x_host: Vec<GoldilocksField> = values.iter().map(|v| GoldilocksField(*v)).collect();
        let stream = CudaStream::default();
        let length = x_host.len();
        let mut result_host = vec![GoldilocksField::ZERO; length];
        let mut x_device = DeviceAllocation::<GoldilocksField>::alloc(length).unwrap();
        let mut result_device = DeviceAllocation::<GoldilocksField>::alloc(length).unwrap();
        memory_copy_async(&mut x_device, &x_host, &stream).unwrap();
        device_fn(&x_device, parameter, &mut result_device, &stream).unwrap();
        memory_copy_async(&mut result_host, &result_device, &stream).unwrap();
        stream.synchronize().unwrap();
        for i in 0..length {
            let left = host_fn(&x_host[i], parameter);
            let right = result_host[i];
            if compare_reduced {
                assert_eq!(left, right);
            } else {
                assert_eq!(left.as_u64(), right.as_u64());
            }
        }
    }

    fn parametrized_unary_op_in_place_test(
        values: &[u64],
        parameter: u32,
        device_fn: ParametrizedUnaryDeviceInPlaceFn,
        host_fn: ParametrizedUnaryHostFn,
        compare_reduced: bool,
    ) {
        let x_host: Vec<GoldilocksField> = values.iter().map(|v| GoldilocksField(*v)).collect();
        let stream = CudaStream::default();
        let length = x_host.len();
        let mut result_host = vec![GoldilocksField::ZERO; length];
        let mut x_device = DeviceAllocation::<GoldilocksField>::alloc(length).unwrap();
        memory_copy_async(&mut x_device, &x_host, &stream).unwrap();
        device_fn(&mut x_device, parameter, &stream).unwrap();
        memory_copy_async(&mut result_host, &x_device, &stream).unwrap();
        stream.synchronize().unwrap();
        for i in 0..length {
            let left = host_fn(&x_host[i], parameter);
            let right = result_host[i];
            if compare_reduced {
                assert_eq!(left, right);
            } else {
                assert_eq!(left.as_u64(), right.as_u64());
            }
        }
    }

    fn binary_op_test(
        x_values: &[u64],
        y_values: &[u64],
        device_fn: BinaryDeviceFn,
        host_fn: BinaryHostFn,
        compare_reduced: bool,
    ) {
        let mut x_host = Vec::<GoldilocksField>::new();
        let mut y_host = Vec::<GoldilocksField>::new();
        x_values
            .iter()
            .cartesian_product(y_values.iter())
            .for_each(|p| {
                x_host.push(GoldilocksField(*p.0));
                y_host.push(GoldilocksField(*p.1));
            });
        let stream = CudaStream::default();
        let length = x_host.len();
        let mut result_host = vec![GoldilocksField::ZERO; length];
        let mut x_device = DeviceAllocation::<GoldilocksField>::alloc(length).unwrap();
        let mut y_device = DeviceAllocation::<GoldilocksField>::alloc(length).unwrap();
        let mut result_device = DeviceAllocation::<GoldilocksField>::alloc(length).unwrap();
        memory_copy_async(&mut x_device, &x_host, &stream).unwrap();
        memory_copy_async(&mut y_device, &y_host, &stream).unwrap();
        device_fn(&x_device, &y_device, &mut result_device, &stream).unwrap();
        memory_copy_async(&mut result_host, &result_device, &stream).unwrap();
        stream.synchronize().unwrap();
        for i in 0..length {
            let left = host_fn(&x_host[i], &y_host[i]);
            let right = result_host[i];
            if compare_reduced {
                assert_eq!(left, right);
            } else {
                assert_eq!(left.as_u64(), right.as_u64());
            }
        }
    }

    fn binary_op_in_place_test(
        x_values: &[u64],
        y_values: &[u64],
        device_fn_mutable_arg_order_standardized: BinaryDeviceInPlaceFn,
        host_fn: BinaryHostFn,
        compare_reduced: bool,
    ) {
        let mut x_host = Vec::<GoldilocksField>::new();
        let mut y_host = Vec::<GoldilocksField>::new();
        x_values
            .iter()
            .cartesian_product(y_values.iter())
            .for_each(|p| {
                x_host.push(GoldilocksField(*p.0));
                y_host.push(GoldilocksField(*p.1));
            });
        let stream = CudaStream::default();
        let length = x_host.len();
        let mut result_host = vec![GoldilocksField::ZERO; length];
        let mut x_device = DeviceAllocation::<GoldilocksField>::alloc(length).unwrap();
        let mut y_device = DeviceAllocation::<GoldilocksField>::alloc(length).unwrap();
        memory_copy_async(&mut x_device, &x_host, &stream).unwrap();
        memory_copy_async(&mut y_device, &y_host, &stream).unwrap();
        device_fn_mutable_arg_order_standardized(&mut x_device, &y_device, &stream).unwrap();
        memory_copy_async(&mut result_host, &x_device, &stream).unwrap();
        stream.synchronize().unwrap();
        for i in 0..length {
            let left = host_fn(&x_host[i], &y_host[i]);
            let right = result_host[i];
            if compare_reduced {
                assert_eq!(left, right);
            } else {
                assert_eq!(left.as_u64(), right.as_u64());
            }
        }
    }

    fn ternary_op_test(
        x_values: &[u64],
        y_values: &[u64],
        z_values: &[u64],
        device_fn: TernaryDeviceFn,
        host_fn: TernaryHostFn,
        compare_reduced: bool,
    ) {
        let length = x_values.len() * y_values.len() * z_values.len();
        let mut x_host = vec![GoldilocksField::ZERO; length];
        let mut y_host = vec![GoldilocksField::ZERO; length];
        let mut z_host = vec![GoldilocksField::ZERO; length];
        // Is there a more Rustaceous way to express this?
        // (ie, analogous to the two-variable cartesian_product, but for three variables)
        let mut idx = 0;
        for &x in x_values {
            for &y in y_values {
                for &z in z_values {
                    x_host[idx] = GoldilocksField(x);
                    y_host[idx] = GoldilocksField(y);
                    z_host[idx] = GoldilocksField(z);
                    idx += 1;
                }
            }
        }
        let stream = CudaStream::default();
        let mut result_host = vec![GoldilocksField::ZERO; length];
        let mut x_device = DeviceAllocation::<GoldilocksField>::alloc(length).unwrap();
        let mut y_device = DeviceAllocation::<GoldilocksField>::alloc(length).unwrap();
        let mut z_device = DeviceAllocation::<GoldilocksField>::alloc(length).unwrap();
        let mut result_device = DeviceAllocation::<GoldilocksField>::alloc(length).unwrap();
        memory_copy_async(&mut x_device, &x_host, &stream).unwrap();
        memory_copy_async(&mut y_device, &y_host, &stream).unwrap();
        memory_copy_async(&mut z_device, &z_host, &stream).unwrap();
        device_fn(&x_device, &y_device, &z_device, &mut result_device, &stream).unwrap();
        memory_copy_async(&mut result_host, &result_device, &stream).unwrap();
        stream.synchronize().unwrap();
        for i in 0..length {
            let left = host_fn(&x_host[i], &y_host[i], &z_host[i]);
            let right = result_host[i];
            if compare_reduced {
                assert_eq!(left, right);
            } else {
                assert_eq!(left.as_u64(), right.as_u64());
            }
        }
    }

    fn ternary_op_in_place_test(
        x_values: &[u64],
        y_values: &[u64],
        z_values: &[u64],
        device_fn: TernaryDeviceInPlaceFn,
        host_fn: TernaryHostFn,
        compare_reduced: bool,
    ) {
        let length = x_values.len() * y_values.len() * z_values.len();
        let mut x_host = vec![GoldilocksField::ZERO; length];
        let mut y_host = vec![GoldilocksField::ZERO; length];
        let mut z_host = vec![GoldilocksField::ZERO; length];
        let mut idx = 0;
        for &x in x_values {
            for &y in y_values {
                for &z in z_values {
                    x_host[idx] = GoldilocksField(x);
                    y_host[idx] = GoldilocksField(y);
                    z_host[idx] = GoldilocksField(z);
                    idx += 1;
                }
            }
        }
        let stream = CudaStream::default();
        let mut result_host = vec![GoldilocksField::ZERO; length];
        let mut x_device = DeviceAllocation::<GoldilocksField>::alloc(length).unwrap();
        let mut y_device = DeviceAllocation::<GoldilocksField>::alloc(length).unwrap();
        let mut z_device = DeviceAllocation::<GoldilocksField>::alloc(length).unwrap();
        memory_copy_async(&mut x_device, &x_host, &stream).unwrap();
        memory_copy_async(&mut y_device, &y_host, &stream).unwrap();
        memory_copy_async(&mut z_device, &z_host, &stream).unwrap();
        device_fn(&mut x_device, &y_device, &z_device, &stream).unwrap();
        memory_copy_async(&mut result_host, &x_device, &stream).unwrap();
        stream.synchronize().unwrap();
        for i in 0..length {
            let left = host_fn(&x_host[i], &y_host[i], &z_host[i]);
            let right = result_host[i];
            if compare_reduced {
                assert_eq!(left, right);
            } else {
                assert_eq!(left.as_u64(), right.as_u64());
            }
        }
    }

    const EPSILON: u64 = (1 << 32) - 1;

    const UNREDUCED_VALUES: [u64; 15] = [
        0,
        1,
        2,
        EPSILON - 2,
        EPSILON - 1,
        EPSILON,
        EPSILON + 1,
        EPSILON + 2,
        GoldilocksField::ORDER - 2,
        GoldilocksField::ORDER - 1,
        GoldilocksField::ORDER,
        GoldilocksField::ORDER + 1,
        GoldilocksField::ORDER + 2,
        u64::MAX - 2,
        u64::MAX - 1,
    ];

    #[test]
    fn dbl() {
        unary_op_test(&UNREDUCED_VALUES, super::dbl, |x| *x.clone().double(), true);
    }

    #[test]
    fn dbl_in_place() {
        unary_op_in_place_test(
            &UNREDUCED_VALUES,
            super::dbl_in_place,
            |x| *x.clone().double(),
            true,
        );
    }

    #[test]
    fn inv() {
        let host_fn = |x: &GoldilocksField| {
            GoldilocksField(x.to_reduced_u64())
                .inverse()
                .unwrap_or(GoldilocksField::ZERO)
        };
        let device_fn = super::inv;
        unary_op_test(&UNREDUCED_VALUES, device_fn, host_fn, true);
    }

    #[test]
    fn inv_in_place() {
        let host_fn = |x: &GoldilocksField| {
            GoldilocksField(x.to_reduced_u64())
                .inverse()
                .unwrap_or(GoldilocksField::ZERO)
        };
        let device_fn = super::inv_in_place;
        unary_op_in_place_test(&UNREDUCED_VALUES, device_fn, host_fn, true);
    }

    #[test]
    fn neg() {
        unary_op_test(&UNREDUCED_VALUES, super::neg, |x| *x.clone().negate(), true);
    }

    #[test]
    fn neg_in_place() {
        unary_op_in_place_test(
            &UNREDUCED_VALUES,
            super::neg_in_place,
            |x| *x.clone().negate(),
            true,
        );
    }

    #[test]
    fn sqr() {
        let host_fn = |x: &GoldilocksField| *x.clone().square();
        let device_fn = super::sqr;
        unary_op_test(&UNREDUCED_VALUES, device_fn, host_fn, true);
    }

    #[test]
    fn sqr_in_place() {
        let host_fn = |x: &GoldilocksField| *x.clone().square();
        let device_fn = super::sqr_in_place;
        unary_op_in_place_test(&UNREDUCED_VALUES, device_fn, host_fn, true);
    }

    #[test]
    fn add() {
        let host_fn = |x: &GoldilocksField, y: &GoldilocksField| x.add(y);
        let device_fn = super::add;
        binary_op_test(
            &UNREDUCED_VALUES,
            &UNREDUCED_VALUES,
            device_fn,
            host_fn,
            true,
        );
    }

    #[test]
    fn add_into_x() {
        let host_fn = |x: &GoldilocksField, y: &GoldilocksField| x.add(y);
        let device_fn = super::add_into_x;
        binary_op_in_place_test(
            &UNREDUCED_VALUES,
            &UNREDUCED_VALUES,
            device_fn,
            host_fn,
            true,
        );
    }

    #[test]
    fn add_into_y() {
        let host_fn = |x: &GoldilocksField, y: &GoldilocksField| x.add(y);
        let device_fn =
            |y: &mut DeviceSlice<GoldilocksField>,
             x: &DeviceSlice<GoldilocksField>,
             stream: &CudaStream| super::add_into_y(x, y, stream);
        binary_op_in_place_test(
            &UNREDUCED_VALUES,
            &UNREDUCED_VALUES,
            device_fn,
            host_fn,
            true,
        );
    }

    #[test]
    fn mul() {
        let host_fn = |x: &GoldilocksField, y: &GoldilocksField| x.mul(y);
        let device_fn = super::mul;
        binary_op_test(
            &UNREDUCED_VALUES,
            &UNREDUCED_VALUES,
            device_fn,
            host_fn,
            true,
        );
    }

    #[test]
    fn mul_into_x() {
        let host_fn = |x: &GoldilocksField, y: &GoldilocksField| x.mul(y);
        let device_fn = super::mul_into_x;
        binary_op_in_place_test(
            &UNREDUCED_VALUES,
            &UNREDUCED_VALUES,
            device_fn,
            host_fn,
            true,
        );
    }

    #[test]
    fn mul_into_y() {
        let host_fn = |x: &GoldilocksField, y: &GoldilocksField| x.mul(y);
        let device_fn =
            |x: &mut DeviceSlice<GoldilocksField>,
             y: &DeviceSlice<GoldilocksField>,
             stream: &CudaStream| super::mul_into_y(y, x, stream);
        binary_op_in_place_test(
            &UNREDUCED_VALUES,
            &UNREDUCED_VALUES,
            device_fn,
            host_fn,
            true,
        );
    }

    #[test]
    fn sub() {
        let host_fn = |x: &GoldilocksField, y: &GoldilocksField| x.sub(y);
        let device_fn = super::sub;
        binary_op_test(
            &UNREDUCED_VALUES,
            &UNREDUCED_VALUES,
            device_fn,
            host_fn,
            true,
        );
    }

    #[test]
    fn sub_into_x() {
        let host_fn = |x: &GoldilocksField, y: &GoldilocksField| x.sub(y);
        let device_fn = super::sub_into_x;
        binary_op_in_place_test(
            &UNREDUCED_VALUES,
            &UNREDUCED_VALUES,
            device_fn,
            host_fn,
            true,
        );
    }

    #[test]
    fn sub_into_y() {
        let host_fn = |y: &GoldilocksField, x: &GoldilocksField| x.sub(y);
        let device_fn =
            |y: &mut DeviceSlice<GoldilocksField>,
             x: &DeviceSlice<GoldilocksField>,
             stream: &CudaStream| super::sub_into_y(x, y, stream);
        binary_op_in_place_test(
            &UNREDUCED_VALUES,
            &UNREDUCED_VALUES,
            device_fn,
            host_fn,
            true,
        );
    }

    #[test]
    fn pow() {
        let host_fn = |x: &GoldilocksField, power: u32| x.pow_u64(power as u64);
        let device_fn = super::pow;
        for power in 0..256 {
            parametrized_unary_op_test(&UNREDUCED_VALUES, power, device_fn, host_fn, true);
        }
    }

    #[test]
    fn pow_in_place() {
        let host_fn = |x: &GoldilocksField, power: u32| x.pow_u64(power as u64);
        let device_fn = super::pow_in_place;
        for power in 0..256 {
            parametrized_unary_op_in_place_test(&UNREDUCED_VALUES, power, device_fn, host_fn, true);
        }
    }

    #[test]
    fn shr() {
        let host_fn = |x: &GoldilocksField, shift: u32| GoldilocksField(x.as_u64() >> shift);
        let device_fn = super::shr;
        parametrized_unary_op_test(&UNREDUCED_VALUES, 0, device_fn, host_fn, false);
        parametrized_unary_op_test(&UNREDUCED_VALUES, 1, device_fn, host_fn, false);
        parametrized_unary_op_test(&UNREDUCED_VALUES, 30, device_fn, host_fn, false);
        parametrized_unary_op_test(&UNREDUCED_VALUES, 31, device_fn, host_fn, false);
    }

    #[test]
    fn shr_in_place() {
        let host_fn = |x: &GoldilocksField, shift: u32| GoldilocksField(x.as_u64() >> shift);
        let device_fn = super::shr_in_place;
        parametrized_unary_op_in_place_test(&UNREDUCED_VALUES, 0, device_fn, host_fn, false);
        parametrized_unary_op_in_place_test(&UNREDUCED_VALUES, 1, device_fn, host_fn, false);
        parametrized_unary_op_in_place_test(&UNREDUCED_VALUES, 30, device_fn, host_fn, false);
        parametrized_unary_op_in_place_test(&UNREDUCED_VALUES, 31, device_fn, host_fn, false);
    }

    #[test]
    fn shl() {
        let host_fn = |x: &GoldilocksField, shift: u32| GoldilocksField(x.as_u64() << shift);
        let device_fn = super::shl;
        parametrized_unary_op_test(&UNREDUCED_VALUES, 0, device_fn, host_fn, false);
        parametrized_unary_op_test(&UNREDUCED_VALUES, 1, device_fn, host_fn, false);
        parametrized_unary_op_test(&UNREDUCED_VALUES, 30, device_fn, host_fn, false);
        parametrized_unary_op_test(&UNREDUCED_VALUES, 31, device_fn, host_fn, false);
    }

    #[test]
    fn shl_in_place() {
        let host_fn = |x: &GoldilocksField, shift: u32| GoldilocksField(x.as_u64() << shift);
        let device_fn = super::shl_in_place;
        parametrized_unary_op_in_place_test(&UNREDUCED_VALUES, 0, device_fn, host_fn, false);
        parametrized_unary_op_in_place_test(&UNREDUCED_VALUES, 1, device_fn, host_fn, false);
        parametrized_unary_op_in_place_test(&UNREDUCED_VALUES, 30, device_fn, host_fn, false);
        parametrized_unary_op_in_place_test(&UNREDUCED_VALUES, 31, device_fn, host_fn, false);
    }

    #[test]
    fn mul_add() {
        let host_fn =
            |x: &GoldilocksField, y: &GoldilocksField, z: &GoldilocksField| x.mul(y).add(z);
        let device_fn = super::mul_add;
        ternary_op_test(
            &UNREDUCED_VALUES,
            &UNREDUCED_VALUES,
            &UNREDUCED_VALUES,
            device_fn,
            host_fn,
            true,
        );
    }

    #[test]
    fn mul_add_into_x() {
        let host_fn =
            |x: &GoldilocksField, y: &GoldilocksField, z: &GoldilocksField| x.mul(y).add(z);
        let device_fn = super::mul_add_into_x;
        ternary_op_in_place_test(
            &UNREDUCED_VALUES,
            &UNREDUCED_VALUES,
            &UNREDUCED_VALUES,
            device_fn,
            host_fn,
            true,
        );
    }

    #[test]
    fn mul_add_into_y() {
        let host_fn =
            |y: &GoldilocksField, x: &GoldilocksField, z: &GoldilocksField| x.mul(y).add(z);
        let device_fn =
            |y: &mut DeviceSlice<GoldilocksField>,
             x: &DeviceSlice<GoldilocksField>,
             z: &DeviceSlice<GoldilocksField>,
             stream: &CudaStream| super::mul_add_into_y(x, y, z, stream);
        ternary_op_in_place_test(
            &UNREDUCED_VALUES,
            &UNREDUCED_VALUES,
            &UNREDUCED_VALUES,
            device_fn,
            host_fn,
            true,
        );
    }

    #[test]
    fn mul_add_into_z() {
        let host_fn =
            |z: &GoldilocksField, x: &GoldilocksField, y: &GoldilocksField| x.mul(y).add(z);
        let device_fn =
            |z: &mut DeviceSlice<GoldilocksField>,
             x: &DeviceSlice<GoldilocksField>,
             y: &DeviceSlice<GoldilocksField>,
             stream: &CudaStream| super::mul_add_into_z(x, y, z, stream);
        ternary_op_in_place_test(
            &UNREDUCED_VALUES,
            &UNREDUCED_VALUES,
            &UNREDUCED_VALUES,
            device_fn,
            host_fn,
            true,
        );
    }

    #[test]
    fn mul_sub() {
        let host_fn =
            |x: &GoldilocksField, y: &GoldilocksField, z: &GoldilocksField| x.mul(y).sub(z);
        let device_fn = super::mul_sub;
        ternary_op_test(
            &UNREDUCED_VALUES,
            &UNREDUCED_VALUES,
            &UNREDUCED_VALUES,
            device_fn,
            host_fn,
            true,
        );
    }

    #[test]
    fn mul_sub_into_x() {
        let host_fn =
            |x: &GoldilocksField, y: &GoldilocksField, z: &GoldilocksField| x.mul(y).sub(z);
        let device_fn = super::mul_sub_into_x;
        ternary_op_in_place_test(
            &UNREDUCED_VALUES,
            &UNREDUCED_VALUES,
            &UNREDUCED_VALUES,
            device_fn,
            host_fn,
            true,
        );
    }

    #[test]
    fn mul_sub_into_y() {
        let host_fn =
            |y: &GoldilocksField, x: &GoldilocksField, z: &GoldilocksField| x.mul(y).sub(z);
        let device_fn =
            |y: &mut DeviceSlice<GoldilocksField>,
             x: &DeviceSlice<GoldilocksField>,
             z: &DeviceSlice<GoldilocksField>,
             stream: &CudaStream| super::mul_sub_into_y(x, y, z, stream);
        ternary_op_in_place_test(
            &UNREDUCED_VALUES,
            &UNREDUCED_VALUES,
            &UNREDUCED_VALUES,
            device_fn,
            host_fn,
            true,
        );
    }

    #[test]
    fn mul_sub_into_z() {
        let host_fn =
            |z: &GoldilocksField, x: &GoldilocksField, y: &GoldilocksField| x.mul(y).sub(z);
        let device_fn =
            |z: &mut DeviceSlice<GoldilocksField>,
             x: &DeviceSlice<GoldilocksField>,
             y: &DeviceSlice<GoldilocksField>,
             stream: &CudaStream| super::mul_sub_into_z(x, y, z, stream);
        ternary_op_in_place_test(
            &UNREDUCED_VALUES,
            &UNREDUCED_VALUES,
            &UNREDUCED_VALUES,
            device_fn,
            host_fn,
            true,
        );
    }
}
