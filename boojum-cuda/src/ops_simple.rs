use cudart::execution::{KernelFourArgs, KernelLaunch, KernelThreeArgs, KernelTwoArgs};
use cudart::memory::memory_set_async;
use cudart::result::CudaResult;
use cudart::slice::DeviceSlice;
use cudart::stream::CudaStream;
use cudart_sys::dim3;

use crate::device_structures::{
    BaseFieldDeviceType, DeviceMatrixChunkImpl, DeviceMatrixChunkMutImpl, DeviceRepr,
    MutPtrAndStrideWrappingMatrix, PtrAndStrideWrappingMatrix, U32DeviceType, U64DeviceType,
    VectorizedExtensionFieldDeviceType,
};
use crate::extension_field::VectorizedExtensionField;
use crate::utils::{get_grid_block_dims_for_threads_count, WARP_SIZE};
use crate::BaseField;

extern "C" {
    fn set_by_val_bf_kernel(
        value: BaseField,
        result: MutPtrAndStrideWrappingMatrix<BaseFieldDeviceType>,
    );

    fn set_by_val_ef_kernel(
        value: VectorizedExtensionField,
        result: MutPtrAndStrideWrappingMatrix<VectorizedExtensionFieldDeviceType>,
    );

    fn set_by_val_u32_kernel(value: u32, result: MutPtrAndStrideWrappingMatrix<U32DeviceType>);

    fn set_by_val_u64_kernel(value: u64, result: MutPtrAndStrideWrappingMatrix<U64DeviceType>);

    fn set_by_ref_bf_kernel(
        values: PtrAndStrideWrappingMatrix<BaseFieldDeviceType>,
        result: MutPtrAndStrideWrappingMatrix<BaseFieldDeviceType>,
    );

    fn set_by_ref_ef_kernel(
        values: PtrAndStrideWrappingMatrix<VectorizedExtensionFieldDeviceType>,
        result: MutPtrAndStrideWrappingMatrix<VectorizedExtensionFieldDeviceType>,
    );

    fn set_by_ref_u32_kernel(
        values: PtrAndStrideWrappingMatrix<U32DeviceType>,
        result: MutPtrAndStrideWrappingMatrix<U32DeviceType>,
    );

    fn set_by_ref_u64_kernel(
        values: PtrAndStrideWrappingMatrix<U64DeviceType>,
        result: MutPtrAndStrideWrappingMatrix<U64DeviceType>,
    );

    fn dbl_bf_kernel(
        x: PtrAndStrideWrappingMatrix<BaseFieldDeviceType>,
        result: MutPtrAndStrideWrappingMatrix<BaseFieldDeviceType>,
    );

    fn dbl_ef_kernel(
        x: PtrAndStrideWrappingMatrix<VectorizedExtensionFieldDeviceType>,
        result: MutPtrAndStrideWrappingMatrix<VectorizedExtensionFieldDeviceType>,
    );

    fn inv_bf_kernel(
        x: PtrAndStrideWrappingMatrix<BaseFieldDeviceType>,
        result: MutPtrAndStrideWrappingMatrix<BaseFieldDeviceType>,
    );

    fn inv_ef_kernel(
        x: PtrAndStrideWrappingMatrix<VectorizedExtensionFieldDeviceType>,
        result: MutPtrAndStrideWrappingMatrix<VectorizedExtensionFieldDeviceType>,
    );

    fn neg_bf_kernel(
        x: PtrAndStrideWrappingMatrix<BaseFieldDeviceType>,
        result: MutPtrAndStrideWrappingMatrix<BaseFieldDeviceType>,
    );

    fn neg_ef_kernel(
        x: PtrAndStrideWrappingMatrix<VectorizedExtensionFieldDeviceType>,
        result: MutPtrAndStrideWrappingMatrix<VectorizedExtensionFieldDeviceType>,
    );

    fn sqr_bf_kernel(
        x: PtrAndStrideWrappingMatrix<BaseFieldDeviceType>,
        result: MutPtrAndStrideWrappingMatrix<BaseFieldDeviceType>,
    );

    fn sqr_ef_kernel(
        x: PtrAndStrideWrappingMatrix<VectorizedExtensionFieldDeviceType>,
        result: MutPtrAndStrideWrappingMatrix<VectorizedExtensionFieldDeviceType>,
    );

    fn shr_kernel(
        x: PtrAndStrideWrappingMatrix<BaseFieldDeviceType>,
        shift: u32,
        result: MutPtrAndStrideWrappingMatrix<BaseFieldDeviceType>,
    );

    fn shl_kernel(
        x: PtrAndStrideWrappingMatrix<BaseFieldDeviceType>,
        shift: u32,
        result: MutPtrAndStrideWrappingMatrix<BaseFieldDeviceType>,
    );

    fn pow_bf_kernel(
        x: PtrAndStrideWrappingMatrix<BaseFieldDeviceType>,
        exponent: u32,
        result: MutPtrAndStrideWrappingMatrix<BaseFieldDeviceType>,
    );

    fn pow_ef_kernel(
        x: PtrAndStrideWrappingMatrix<VectorizedExtensionFieldDeviceType>,
        exponent: u32,
        result: MutPtrAndStrideWrappingMatrix<VectorizedExtensionFieldDeviceType>,
    );

    fn add_bf_bf_kernel(
        x: PtrAndStrideWrappingMatrix<BaseFieldDeviceType>,
        y: PtrAndStrideWrappingMatrix<BaseFieldDeviceType>,
        result: MutPtrAndStrideWrappingMatrix<BaseFieldDeviceType>,
    );

    fn add_bf_ef_kernel(
        x: PtrAndStrideWrappingMatrix<BaseFieldDeviceType>,
        y: PtrAndStrideWrappingMatrix<VectorizedExtensionFieldDeviceType>,
        result: MutPtrAndStrideWrappingMatrix<VectorizedExtensionFieldDeviceType>,
    );

    fn add_ef_bf_kernel(
        x: PtrAndStrideWrappingMatrix<VectorizedExtensionFieldDeviceType>,
        y: PtrAndStrideWrappingMatrix<BaseFieldDeviceType>,
        result: MutPtrAndStrideWrappingMatrix<VectorizedExtensionFieldDeviceType>,
    );

    fn add_ef_ef_kernel(
        x: PtrAndStrideWrappingMatrix<VectorizedExtensionFieldDeviceType>,
        y: PtrAndStrideWrappingMatrix<VectorizedExtensionFieldDeviceType>,
        result: MutPtrAndStrideWrappingMatrix<VectorizedExtensionFieldDeviceType>,
    );

    fn mul_bf_bf_kernel(
        x: PtrAndStrideWrappingMatrix<BaseFieldDeviceType>,
        y: PtrAndStrideWrappingMatrix<BaseFieldDeviceType>,
        result: MutPtrAndStrideWrappingMatrix<BaseFieldDeviceType>,
    );

    fn mul_bf_ef_kernel(
        x: PtrAndStrideWrappingMatrix<BaseFieldDeviceType>,
        y: PtrAndStrideWrappingMatrix<VectorizedExtensionFieldDeviceType>,
        result: MutPtrAndStrideWrappingMatrix<VectorizedExtensionFieldDeviceType>,
    );

    fn mul_ef_bf_kernel(
        x: PtrAndStrideWrappingMatrix<VectorizedExtensionFieldDeviceType>,
        y: PtrAndStrideWrappingMatrix<BaseFieldDeviceType>,
        result: MutPtrAndStrideWrappingMatrix<VectorizedExtensionFieldDeviceType>,
    );

    fn mul_ef_ef_kernel(
        x: PtrAndStrideWrappingMatrix<VectorizedExtensionFieldDeviceType>,
        y: PtrAndStrideWrappingMatrix<VectorizedExtensionFieldDeviceType>,
        result: MutPtrAndStrideWrappingMatrix<VectorizedExtensionFieldDeviceType>,
    );

    fn sub_bf_bf_kernel(
        x: PtrAndStrideWrappingMatrix<BaseFieldDeviceType>,
        y: PtrAndStrideWrappingMatrix<BaseFieldDeviceType>,
        result: MutPtrAndStrideWrappingMatrix<BaseFieldDeviceType>,
    );

    fn sub_bf_ef_kernel(
        x: PtrAndStrideWrappingMatrix<BaseFieldDeviceType>,
        y: PtrAndStrideWrappingMatrix<VectorizedExtensionFieldDeviceType>,
        result: MutPtrAndStrideWrappingMatrix<VectorizedExtensionFieldDeviceType>,
    );

    fn sub_ef_bf_kernel(
        x: PtrAndStrideWrappingMatrix<VectorizedExtensionFieldDeviceType>,
        y: PtrAndStrideWrappingMatrix<BaseFieldDeviceType>,
        result: MutPtrAndStrideWrappingMatrix<VectorizedExtensionFieldDeviceType>,
    );

    fn sub_ef_ef_kernel(
        x: PtrAndStrideWrappingMatrix<VectorizedExtensionFieldDeviceType>,
        y: PtrAndStrideWrappingMatrix<VectorizedExtensionFieldDeviceType>,
        result: MutPtrAndStrideWrappingMatrix<VectorizedExtensionFieldDeviceType>,
    );

    fn mul_add_bf_bf_bf_kernel(
        x: PtrAndStrideWrappingMatrix<BaseFieldDeviceType>,
        y: PtrAndStrideWrappingMatrix<BaseFieldDeviceType>,
        z: PtrAndStrideWrappingMatrix<BaseFieldDeviceType>,
        result: MutPtrAndStrideWrappingMatrix<BaseFieldDeviceType>,
    );

    fn mul_add_bf_bf_ef_kernel(
        x: PtrAndStrideWrappingMatrix<BaseFieldDeviceType>,
        y: PtrAndStrideWrappingMatrix<BaseFieldDeviceType>,
        z: PtrAndStrideWrappingMatrix<VectorizedExtensionFieldDeviceType>,
        result: MutPtrAndStrideWrappingMatrix<VectorizedExtensionFieldDeviceType>,
    );

    fn mul_add_bf_ef_bf_kernel(
        x: PtrAndStrideWrappingMatrix<BaseFieldDeviceType>,
        y: PtrAndStrideWrappingMatrix<VectorizedExtensionFieldDeviceType>,
        z: PtrAndStrideWrappingMatrix<BaseFieldDeviceType>,
        result: MutPtrAndStrideWrappingMatrix<VectorizedExtensionFieldDeviceType>,
    );

    fn mul_add_bf_ef_ef_kernel(
        x: PtrAndStrideWrappingMatrix<BaseFieldDeviceType>,
        y: PtrAndStrideWrappingMatrix<VectorizedExtensionFieldDeviceType>,
        z: PtrAndStrideWrappingMatrix<VectorizedExtensionFieldDeviceType>,
        result: MutPtrAndStrideWrappingMatrix<VectorizedExtensionFieldDeviceType>,
    );

    fn mul_add_ef_bf_bf_kernel(
        x: PtrAndStrideWrappingMatrix<VectorizedExtensionFieldDeviceType>,
        y: PtrAndStrideWrappingMatrix<BaseFieldDeviceType>,
        z: PtrAndStrideWrappingMatrix<BaseFieldDeviceType>,
        result: MutPtrAndStrideWrappingMatrix<VectorizedExtensionFieldDeviceType>,
    );

    fn mul_add_ef_bf_ef_kernel(
        x: PtrAndStrideWrappingMatrix<VectorizedExtensionFieldDeviceType>,
        y: PtrAndStrideWrappingMatrix<BaseFieldDeviceType>,
        z: PtrAndStrideWrappingMatrix<VectorizedExtensionFieldDeviceType>,
        result: MutPtrAndStrideWrappingMatrix<VectorizedExtensionFieldDeviceType>,
    );

    fn mul_add_ef_ef_bf_kernel(
        x: PtrAndStrideWrappingMatrix<VectorizedExtensionFieldDeviceType>,
        y: PtrAndStrideWrappingMatrix<VectorizedExtensionFieldDeviceType>,
        z: PtrAndStrideWrappingMatrix<BaseFieldDeviceType>,
        result: MutPtrAndStrideWrappingMatrix<VectorizedExtensionFieldDeviceType>,
    );

    fn mul_add_ef_ef_ef_kernel(
        x: PtrAndStrideWrappingMatrix<VectorizedExtensionFieldDeviceType>,
        y: PtrAndStrideWrappingMatrix<VectorizedExtensionFieldDeviceType>,
        z: PtrAndStrideWrappingMatrix<VectorizedExtensionFieldDeviceType>,
        result: MutPtrAndStrideWrappingMatrix<VectorizedExtensionFieldDeviceType>,
    );

    fn mul_sub_bf_bf_bf_kernel(
        x: PtrAndStrideWrappingMatrix<BaseFieldDeviceType>,
        y: PtrAndStrideWrappingMatrix<BaseFieldDeviceType>,
        z: PtrAndStrideWrappingMatrix<BaseFieldDeviceType>,
        result: MutPtrAndStrideWrappingMatrix<BaseFieldDeviceType>,
    );

    fn mul_sub_bf_bf_ef_kernel(
        x: PtrAndStrideWrappingMatrix<BaseFieldDeviceType>,
        y: PtrAndStrideWrappingMatrix<BaseFieldDeviceType>,
        z: PtrAndStrideWrappingMatrix<VectorizedExtensionFieldDeviceType>,
        result: MutPtrAndStrideWrappingMatrix<VectorizedExtensionFieldDeviceType>,
    );

    fn mul_sub_bf_ef_bf_kernel(
        x: PtrAndStrideWrappingMatrix<BaseFieldDeviceType>,
        y: PtrAndStrideWrappingMatrix<VectorizedExtensionFieldDeviceType>,
        z: PtrAndStrideWrappingMatrix<BaseFieldDeviceType>,
        result: MutPtrAndStrideWrappingMatrix<VectorizedExtensionFieldDeviceType>,
    );

    fn mul_sub_bf_ef_ef_kernel(
        x: PtrAndStrideWrappingMatrix<BaseFieldDeviceType>,
        y: PtrAndStrideWrappingMatrix<VectorizedExtensionFieldDeviceType>,
        z: PtrAndStrideWrappingMatrix<VectorizedExtensionFieldDeviceType>,
        result: MutPtrAndStrideWrappingMatrix<VectorizedExtensionFieldDeviceType>,
    );

    fn mul_sub_ef_bf_bf_kernel(
        x: PtrAndStrideWrappingMatrix<VectorizedExtensionFieldDeviceType>,
        y: PtrAndStrideWrappingMatrix<BaseFieldDeviceType>,
        z: PtrAndStrideWrappingMatrix<BaseFieldDeviceType>,
        result: MutPtrAndStrideWrappingMatrix<VectorizedExtensionFieldDeviceType>,
    );

    fn mul_sub_ef_bf_ef_kernel(
        x: PtrAndStrideWrappingMatrix<VectorizedExtensionFieldDeviceType>,
        y: PtrAndStrideWrappingMatrix<BaseFieldDeviceType>,
        z: PtrAndStrideWrappingMatrix<VectorizedExtensionFieldDeviceType>,
        result: MutPtrAndStrideWrappingMatrix<VectorizedExtensionFieldDeviceType>,
    );

    fn mul_sub_ef_ef_bf_kernel(
        x: PtrAndStrideWrappingMatrix<VectorizedExtensionFieldDeviceType>,
        y: PtrAndStrideWrappingMatrix<VectorizedExtensionFieldDeviceType>,
        z: PtrAndStrideWrappingMatrix<BaseFieldDeviceType>,
        result: MutPtrAndStrideWrappingMatrix<VectorizedExtensionFieldDeviceType>,
    );

    fn mul_sub_ef_ef_ef_kernel(
        x: PtrAndStrideWrappingMatrix<VectorizedExtensionFieldDeviceType>,
        y: PtrAndStrideWrappingMatrix<VectorizedExtensionFieldDeviceType>,
        z: PtrAndStrideWrappingMatrix<VectorizedExtensionFieldDeviceType>,
        result: MutPtrAndStrideWrappingMatrix<VectorizedExtensionFieldDeviceType>,
    );
}

pub fn set_to_zero<T>(result: &mut DeviceSlice<T>, stream: &CudaStream) -> CudaResult<()> {
    memory_set_async(unsafe { result.transmute_mut() }, 0, stream)
}

fn get_launch_dims(rows: u32, cols: u32) -> (dim3, dim3) {
    let (mut grid_dim, block_dim) = get_grid_block_dims_for_threads_count(WARP_SIZE * 4, rows);
    grid_dim.y = cols;
    (grid_dim, block_dim)
}

pub type SetByValKernel<T> =
    KernelTwoArgs<T, MutPtrAndStrideWrappingMatrix<<T as DeviceRepr>::Type>>;

pub trait SetByVal: Sized + DeviceRepr {
    fn get_kernel() -> SetByValKernel<Self>;

    fn launch(
        value: Self,
        result: &mut (impl DeviceMatrixChunkMutImpl<Self> + ?Sized),
        stream: &CudaStream,
    ) -> CudaResult<()> {
        let result = MutPtrAndStrideWrappingMatrix::new(result);
        let kernel = Self::get_kernel();
        let (grid_dim, block_dim) = get_launch_dims(result.rows, result.cols);
        let args = (&value, &result);
        unsafe { kernel.launch(grid_dim, block_dim, args, 0, stream) }
    }
}

pub fn set_by_val<T: SetByVal>(
    value: T,
    result: &mut (impl DeviceMatrixChunkMutImpl<T> + ?Sized),
    stream: &CudaStream,
) -> CudaResult<()> {
    T::launch(value, result, stream)
}

impl SetByVal for u32 {
    fn get_kernel() -> SetByValKernel<Self> {
        set_by_val_u32_kernel
    }
}

impl SetByVal for u64 {
    fn get_kernel() -> SetByValKernel<Self> {
        set_by_val_u64_kernel
    }
}

impl SetByVal for BaseField {
    fn get_kernel() -> SetByValKernel<Self> {
        set_by_val_bf_kernel
    }
}

impl SetByVal for VectorizedExtensionField {
    fn get_kernel() -> SetByValKernel<Self> {
        set_by_val_ef_kernel
    }
}

pub type SetByRefKernel<T> = KernelTwoArgs<
    PtrAndStrideWrappingMatrix<<T as DeviceRepr>::Type>,
    MutPtrAndStrideWrappingMatrix<<T as DeviceRepr>::Type>,
>;

pub trait SetByRef: Sized + DeviceRepr {
    fn get_kernel() -> SetByRefKernel<Self>;

    fn launch(
        values: &(impl DeviceMatrixChunkImpl<Self> + ?Sized),
        result: &mut (impl DeviceMatrixChunkMutImpl<Self> + ?Sized),
        stream: &CudaStream,
    ) -> CudaResult<()> {
        assert_eq!(result.rows() % values.rows(), 0);
        assert_eq!(result.cols() % values.cols(), 0);
        let values = PtrAndStrideWrappingMatrix::new(values);
        let result = MutPtrAndStrideWrappingMatrix::new(result);
        let kernel = Self::get_kernel();
        let (grid_dim, block_dim) = get_launch_dims(result.rows, result.cols);
        let args = (&values, &result);
        unsafe { kernel.launch(grid_dim, block_dim, args, 0, stream) }
    }
}

pub fn set_by_ref<T: SetByRef>(
    values: &(impl DeviceMatrixChunkImpl<T> + ?Sized),
    result: &mut (impl DeviceMatrixChunkMutImpl<T> + ?Sized),
    stream: &CudaStream,
) -> CudaResult<()> {
    T::launch(values, result, stream)
}

impl SetByRef for u32 {
    fn get_kernel() -> SetByRefKernel<Self> {
        set_by_ref_u32_kernel
    }
}

impl SetByRef for u64 {
    fn get_kernel() -> SetByRefKernel<Self> {
        set_by_ref_u64_kernel
    }
}

impl SetByRef for BaseField {
    fn get_kernel() -> SetByRefKernel<Self> {
        set_by_ref_bf_kernel
    }
}

impl SetByRef for VectorizedExtensionField {
    fn get_kernel() -> SetByRefKernel<Self> {
        set_by_ref_ef_kernel
    }
}

pub enum UnaryOpType {
    Dbl,
    Inv,
    Neg,
    Sqr,
}

type UnaryKernel<T0, TR> = KernelTwoArgs<
    PtrAndStrideWrappingMatrix<<T0 as DeviceRepr>::Type>,
    MutPtrAndStrideWrappingMatrix<<TR as DeviceRepr>::Type>,
>;

pub trait UnaryOp: Sized + DeviceRepr {
    fn get_kernel(op_type: UnaryOpType) -> UnaryKernel<Self, Self>;

    fn launch_op(
        op_type: UnaryOpType,
        x: PtrAndStrideWrappingMatrix<Self::Type>,
        result: MutPtrAndStrideWrappingMatrix<Self::Type>,
        stream: &CudaStream,
    ) -> CudaResult<()> {
        let kernel = Self::get_kernel(op_type);
        let (grid_dim, block_dim) = get_launch_dims(result.rows, result.cols);
        let args = (&x, &result);
        unsafe { kernel.launch(grid_dim, block_dim, args, 0, stream) }
    }

    fn launch(
        op_type: UnaryOpType,
        x: &(impl DeviceMatrixChunkImpl<Self> + ?Sized),
        result: &mut (impl DeviceMatrixChunkMutImpl<Self> + ?Sized),
        stream: &CudaStream,
    ) -> CudaResult<()> {
        assert_eq!(result.rows() % x.rows(), 0);
        assert_eq!(result.cols() % x.cols(), 0);
        Self::launch_op(
            op_type,
            PtrAndStrideWrappingMatrix::new(x),
            MutPtrAndStrideWrappingMatrix::new(result),
            stream,
        )
    }

    fn launch_in_place(
        op_type: UnaryOpType,
        x: &mut (impl DeviceMatrixChunkMutImpl<Self> + ?Sized),
        stream: &CudaStream,
    ) -> CudaResult<()> {
        Self::launch_op(
            op_type,
            PtrAndStrideWrappingMatrix::new(x),
            MutPtrAndStrideWrappingMatrix::new(x),
            stream,
        )
    }
}

impl UnaryOp for BaseField {
    fn get_kernel(op_type: UnaryOpType) -> UnaryKernel<Self, Self> {
        match op_type {
            UnaryOpType::Dbl => dbl_bf_kernel,
            UnaryOpType::Inv => inv_bf_kernel,
            UnaryOpType::Neg => neg_bf_kernel,
            UnaryOpType::Sqr => sqr_bf_kernel,
        }
    }
}

impl UnaryOp for VectorizedExtensionField {
    fn get_kernel(op_type: UnaryOpType) -> UnaryKernel<Self, Self> {
        match op_type {
            UnaryOpType::Dbl => dbl_ef_kernel,
            UnaryOpType::Inv => inv_ef_kernel,
            UnaryOpType::Neg => neg_ef_kernel,
            UnaryOpType::Sqr => sqr_ef_kernel,
        }
    }
}

fn unary_op<T: UnaryOp>(
    op_type: UnaryOpType,
    x: &(impl DeviceMatrixChunkImpl<T> + ?Sized),
    result: &mut (impl DeviceMatrixChunkMutImpl<T> + ?Sized),
    stream: &CudaStream,
) -> CudaResult<()> {
    T::launch(op_type, x, result, stream)
}

fn unary_op_in_place<T: UnaryOp>(
    op_type: UnaryOpType,
    x: &mut (impl DeviceMatrixChunkMutImpl<T> + ?Sized),
    stream: &CudaStream,
) -> CudaResult<()> {
    T::launch_in_place(op_type, x, stream)
}

pub fn dbl<T: UnaryOp>(
    x: &(impl DeviceMatrixChunkImpl<T> + ?Sized),
    result: &mut (impl DeviceMatrixChunkMutImpl<T> + ?Sized),
    stream: &CudaStream,
) -> CudaResult<()> {
    unary_op(UnaryOpType::Dbl, x, result, stream)
}

pub fn dbl_in_place<T: UnaryOp>(
    x: &mut (impl DeviceMatrixChunkMutImpl<T> + ?Sized),
    stream: &CudaStream,
) -> CudaResult<()> {
    unary_op_in_place(UnaryOpType::Dbl, x, stream)
}

pub fn inv<T: UnaryOp>(
    x: &(impl DeviceMatrixChunkImpl<T> + ?Sized),
    result: &mut (impl DeviceMatrixChunkMutImpl<T> + ?Sized),
    stream: &CudaStream,
) -> CudaResult<()> {
    unary_op(UnaryOpType::Inv, x, result, stream)
}

pub fn inv_in_place<T: UnaryOp>(
    x: &mut (impl DeviceMatrixChunkMutImpl<T> + ?Sized),
    stream: &CudaStream,
) -> CudaResult<()> {
    unary_op_in_place(UnaryOpType::Inv, x, stream)
}

pub fn neg<T: UnaryOp>(
    x: &(impl DeviceMatrixChunkImpl<T> + ?Sized),
    result: &mut (impl DeviceMatrixChunkMutImpl<T> + ?Sized),
    stream: &CudaStream,
) -> CudaResult<()> {
    unary_op(UnaryOpType::Neg, x, result, stream)
}

pub fn neg_in_place<T: UnaryOp>(
    x: &mut (impl DeviceMatrixChunkMutImpl<T> + ?Sized),
    stream: &CudaStream,
) -> CudaResult<()> {
    unary_op_in_place(UnaryOpType::Neg, x, stream)
}

pub fn sqr<T: UnaryOp>(
    x: &(impl DeviceMatrixChunkImpl<T> + ?Sized),
    result: &mut (impl DeviceMatrixChunkMutImpl<T> + ?Sized),
    stream: &CudaStream,
) -> CudaResult<()> {
    unary_op(UnaryOpType::Sqr, x, result, stream)
}

pub fn sqr_in_place<T: UnaryOp>(
    x: &mut (impl DeviceMatrixChunkMutImpl<T> + ?Sized),
    stream: &CudaStream,
) -> CudaResult<()> {
    unary_op_in_place(UnaryOpType::Sqr, x, stream)
}

pub enum ParametrizedUnaryOpType {
    Pow,
    Shl,
    Shr,
}

type ParametrizedUnaryKernel<T0, TR> = KernelThreeArgs<
    PtrAndStrideWrappingMatrix<<T0 as DeviceRepr>::Type>,
    u32,
    MutPtrAndStrideWrappingMatrix<<TR as DeviceRepr>::Type>,
>;

pub trait ParametrizedUnaryOp: Sized + DeviceRepr {
    fn get_kernel(op_type: ParametrizedUnaryOpType) -> ParametrizedUnaryKernel<Self, Self>;

    fn launch_op(
        op_type: ParametrizedUnaryOpType,
        x: PtrAndStrideWrappingMatrix<Self::Type>,
        param: u32,
        result: MutPtrAndStrideWrappingMatrix<Self::Type>,
        stream: &CudaStream,
    ) -> CudaResult<()> {
        let kernel = Self::get_kernel(op_type);
        let (grid_dim, block_dim) = get_launch_dims(result.rows, result.cols);
        let args = (&x, &param, &result);
        unsafe { kernel.launch(grid_dim, block_dim, args, 0, stream) }
    }

    fn launch(
        op_type: ParametrizedUnaryOpType,
        x: &(impl DeviceMatrixChunkImpl<Self> + ?Sized),
        param: u32,
        result: &mut (impl DeviceMatrixChunkMutImpl<Self> + ?Sized),
        stream: &CudaStream,
    ) -> CudaResult<()> {
        assert_eq!(result.rows() % x.rows(), 0);
        assert_eq!(result.cols() % x.cols(), 0);
        Self::launch_op(
            op_type,
            PtrAndStrideWrappingMatrix::new(x),
            param,
            MutPtrAndStrideWrappingMatrix::new(result),
            stream,
        )
    }

    fn launch_in_place(
        op_type: ParametrizedUnaryOpType,
        x: &mut (impl DeviceMatrixChunkMutImpl<Self> + ?Sized),
        param: u32,
        stream: &CudaStream,
    ) -> CudaResult<()> {
        Self::launch_op(
            op_type,
            PtrAndStrideWrappingMatrix::new(x),
            param,
            MutPtrAndStrideWrappingMatrix::new(x),
            stream,
        )
    }
}

impl ParametrizedUnaryOp for BaseField {
    fn get_kernel(op_type: ParametrizedUnaryOpType) -> ParametrizedUnaryKernel<Self, Self> {
        match op_type {
            ParametrizedUnaryOpType::Pow => pow_bf_kernel,
            ParametrizedUnaryOpType::Shl => shl_kernel,
            ParametrizedUnaryOpType::Shr => shr_kernel,
        }
    }
}

impl ParametrizedUnaryOp for VectorizedExtensionField {
    fn get_kernel(op_type: ParametrizedUnaryOpType) -> ParametrizedUnaryKernel<Self, Self> {
        match op_type {
            ParametrizedUnaryOpType::Pow => pow_ef_kernel,
            ParametrizedUnaryOpType::Shl => unimplemented!(),
            ParametrizedUnaryOpType::Shr => unimplemented!(),
        }
    }
}

fn parametrized_unary_op<T: ParametrizedUnaryOp>(
    op_type: ParametrizedUnaryOpType,
    x: &(impl DeviceMatrixChunkImpl<T> + ?Sized),
    param: u32,
    result: &mut (impl DeviceMatrixChunkMutImpl<T> + ?Sized),
    stream: &CudaStream,
) -> CudaResult<()> {
    T::launch(op_type, x, param, result, stream)
}

fn parametrized_unary_op_in_place<T: ParametrizedUnaryOp>(
    op_type: ParametrizedUnaryOpType,
    x: &mut (impl DeviceMatrixChunkMutImpl<T> + ?Sized),
    param: u32,
    stream: &CudaStream,
) -> CudaResult<()> {
    T::launch_in_place(op_type, x, param, stream)
}

pub fn pow<T: ParametrizedUnaryOp>(
    x: &(impl DeviceMatrixChunkImpl<T> + ?Sized),
    exponent: u32,
    result: &mut (impl DeviceMatrixChunkMutImpl<T> + ?Sized),
    stream: &CudaStream,
) -> CudaResult<()> {
    parametrized_unary_op(ParametrizedUnaryOpType::Pow, x, exponent, result, stream)
}

pub fn pow_in_place<T: ParametrizedUnaryOp>(
    x: &mut (impl DeviceMatrixChunkMutImpl<T> + ?Sized),
    exponent: u32,
    stream: &CudaStream,
) -> CudaResult<()> {
    parametrized_unary_op_in_place(ParametrizedUnaryOpType::Pow, x, exponent, stream)
}

pub fn shl(
    x: &(impl DeviceMatrixChunkImpl<BaseField> + ?Sized),
    shift: u32,
    result: &mut (impl DeviceMatrixChunkMutImpl<BaseField> + ?Sized),
    stream: &CudaStream,
) -> CudaResult<()> {
    parametrized_unary_op(ParametrizedUnaryOpType::Shl, x, shift, result, stream)
}

pub fn shl_in_place(
    x: &mut (impl DeviceMatrixChunkMutImpl<BaseField> + ?Sized),
    shift: u32,
    stream: &CudaStream,
) -> CudaResult<()> {
    parametrized_unary_op_in_place(ParametrizedUnaryOpType::Shl, x, shift, stream)
}

pub fn shr(
    x: &(impl DeviceMatrixChunkImpl<BaseField> + ?Sized),
    shift: u32,
    result: &mut (impl DeviceMatrixChunkMutImpl<BaseField> + ?Sized),
    stream: &CudaStream,
) -> CudaResult<()> {
    parametrized_unary_op(ParametrizedUnaryOpType::Shr, x, shift, result, stream)
}

pub fn shr_in_place(
    x: &mut (impl DeviceMatrixChunkMutImpl<BaseField> + ?Sized),
    shift: u32,
    stream: &CudaStream,
) -> CudaResult<()> {
    parametrized_unary_op_in_place(ParametrizedUnaryOpType::Shr, x, shift, stream)
}

pub enum BinaryOpType {
    Add,
    Mul,
    Sub,
}

type BinaryKernel<T0, T1, TR> = KernelThreeArgs<
    PtrAndStrideWrappingMatrix<<T0 as DeviceRepr>::Type>,
    PtrAndStrideWrappingMatrix<<T1 as DeviceRepr>::Type>,
    MutPtrAndStrideWrappingMatrix<<TR as DeviceRepr>::Type>,
>;

pub trait BinaryOp<T0: Sized + DeviceRepr, T1: Sized + DeviceRepr, TR: Sized + DeviceRepr> {
    fn get_kernel(op_type: BinaryOpType) -> BinaryKernel<T0, T1, TR>;

    fn launch_op(
        op_type: BinaryOpType,
        x: PtrAndStrideWrappingMatrix<T0::Type>,
        y: PtrAndStrideWrappingMatrix<T1::Type>,
        result: MutPtrAndStrideWrappingMatrix<TR::Type>,
        stream: &CudaStream,
    ) -> CudaResult<()> {
        let kernel = Self::get_kernel(op_type);
        let (grid_dim, block_dim) = get_launch_dims(result.rows, result.cols);
        let args = (&x, &y, &result);
        unsafe { kernel.launch(grid_dim, block_dim, args, 0, stream) }
    }

    fn launch(
        op_type: BinaryOpType,
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
            op_type,
            PtrAndStrideWrappingMatrix::new(x),
            PtrAndStrideWrappingMatrix::new(y),
            MutPtrAndStrideWrappingMatrix::new(result),
            stream,
        )
    }

    fn launch_into_x(
        op_type: BinaryOpType,
        x: &mut (impl DeviceMatrixChunkMutImpl<T0> + ?Sized),
        y: &(impl DeviceMatrixChunkImpl<T1> + ?Sized),
        stream: &CudaStream,
    ) -> CudaResult<()>
    where
        (T0, T1, T0): BinaryOp<T0, T1, T0>,
    {
        <(T0, T1, T0)>::launch_op(
            op_type,
            PtrAndStrideWrappingMatrix::new(x),
            PtrAndStrideWrappingMatrix::new(y),
            MutPtrAndStrideWrappingMatrix::new(x),
            stream,
        )
    }

    fn launch_into_y(
        op_type: BinaryOpType,
        x: &(impl DeviceMatrixChunkImpl<T0> + ?Sized),
        y: &mut (impl DeviceMatrixChunkMutImpl<T1> + ?Sized),
        stream: &CudaStream,
    ) -> CudaResult<()>
    where
        (T0, T1, T1): BinaryOp<T0, T1, T1>,
    {
        <(T0, T1, T1)>::launch_op(
            op_type,
            PtrAndStrideWrappingMatrix::new(x),
            PtrAndStrideWrappingMatrix::new(y),
            MutPtrAndStrideWrappingMatrix::new(y),
            stream,
        )
    }
}

impl BinaryOp<BaseField, BaseField, BaseField> for (BaseField, BaseField, BaseField) {
    fn get_kernel(op_type: BinaryOpType) -> BinaryKernel<BaseField, BaseField, BaseField> {
        match op_type {
            BinaryOpType::Add => add_bf_bf_kernel,
            BinaryOpType::Mul => mul_bf_bf_kernel,
            BinaryOpType::Sub => sub_bf_bf_kernel,
        }
    }
}

impl BinaryOp<BaseField, VectorizedExtensionField, VectorizedExtensionField>
    for (
        BaseField,
        VectorizedExtensionField,
        VectorizedExtensionField,
    )
{
    fn get_kernel(
        op_type: BinaryOpType,
    ) -> BinaryKernel<BaseField, VectorizedExtensionField, VectorizedExtensionField> {
        match op_type {
            BinaryOpType::Add => add_bf_ef_kernel,
            BinaryOpType::Mul => mul_bf_ef_kernel,
            BinaryOpType::Sub => sub_bf_ef_kernel,
        }
    }
}

impl BinaryOp<VectorizedExtensionField, BaseField, VectorizedExtensionField>
    for (
        VectorizedExtensionField,
        BaseField,
        VectorizedExtensionField,
    )
{
    fn get_kernel(
        op_type: BinaryOpType,
    ) -> BinaryKernel<VectorizedExtensionField, BaseField, VectorizedExtensionField> {
        match op_type {
            BinaryOpType::Add => add_ef_bf_kernel,
            BinaryOpType::Mul => mul_ef_bf_kernel,
            BinaryOpType::Sub => sub_ef_bf_kernel,
        }
    }
}

impl BinaryOp<VectorizedExtensionField, VectorizedExtensionField, VectorizedExtensionField>
    for (
        VectorizedExtensionField,
        VectorizedExtensionField,
        VectorizedExtensionField,
    )
{
    fn get_kernel(
        op_type: BinaryOpType,
    ) -> BinaryKernel<VectorizedExtensionField, VectorizedExtensionField, VectorizedExtensionField>
    {
        match op_type {
            BinaryOpType::Add => add_ef_ef_kernel,
            BinaryOpType::Mul => mul_ef_ef_kernel,
            BinaryOpType::Sub => sub_ef_ef_kernel,
        }
    }
}

fn binary_op<T0: DeviceRepr, T1: DeviceRepr, TR: DeviceRepr>(
    op_type: BinaryOpType,
    x: &(impl DeviceMatrixChunkImpl<T0> + ?Sized),
    y: &(impl DeviceMatrixChunkImpl<T1> + ?Sized),
    result: &mut (impl DeviceMatrixChunkMutImpl<TR> + ?Sized),
    stream: &CudaStream,
) -> CudaResult<()>
where
    (T0, T1, TR): BinaryOp<T0, T1, TR>,
{
    <(T0, T1, TR)>::launch(op_type, x, y, result, stream)
}

fn binary_op_into_x<T0: DeviceRepr, T1: DeviceRepr>(
    op_type: BinaryOpType,
    x: &mut (impl DeviceMatrixChunkMutImpl<T0> + ?Sized),
    y: &(impl DeviceMatrixChunkImpl<T1> + ?Sized),
    stream: &CudaStream,
) -> CudaResult<()>
where
    (T0, T1, T0): BinaryOp<T0, T1, T0>,
{
    <(T0, T1, T0)>::launch_into_x(op_type, x, y, stream)
}

fn binary_op_into_y<T0: DeviceRepr, T1: DeviceRepr>(
    op_type: BinaryOpType,
    x: &(impl DeviceMatrixChunkImpl<T0> + ?Sized),
    y: &mut (impl DeviceMatrixChunkMutImpl<T1> + ?Sized),
    stream: &CudaStream,
) -> CudaResult<()>
where
    (T0, T1, T1): BinaryOp<T0, T1, T1>,
{
    <(T0, T1, T1)>::launch_into_y(op_type, x, y, stream)
}

pub fn add<T0: DeviceRepr, T1: DeviceRepr, TR: DeviceRepr>(
    x: &(impl DeviceMatrixChunkImpl<T0> + ?Sized),
    y: &(impl DeviceMatrixChunkImpl<T1> + ?Sized),
    result: &mut (impl DeviceMatrixChunkMutImpl<TR> + ?Sized),
    stream: &CudaStream,
) -> CudaResult<()>
where
    (T0, T1, TR): BinaryOp<T0, T1, TR>,
{
    binary_op(BinaryOpType::Add, x, y, result, stream)
}

pub fn add_into_x<T0: DeviceRepr, T1: DeviceRepr>(
    x: &mut (impl DeviceMatrixChunkMutImpl<T0> + ?Sized),
    y: &(impl DeviceMatrixChunkImpl<T1> + ?Sized),
    stream: &CudaStream,
) -> CudaResult<()>
where
    (T0, T1, T0): BinaryOp<T0, T1, T0>,
{
    binary_op_into_x(BinaryOpType::Add, x, y, stream)
}

pub fn add_into_y<T0: DeviceRepr, T1: DeviceRepr>(
    x: &(impl DeviceMatrixChunkImpl<T0> + ?Sized),
    y: &mut (impl DeviceMatrixChunkMutImpl<T1> + ?Sized),
    stream: &CudaStream,
) -> CudaResult<()>
where
    (T0, T1, T1): BinaryOp<T0, T1, T1>,
{
    binary_op_into_y(BinaryOpType::Add, x, y, stream)
}

pub fn mul<T0: DeviceRepr, T1: DeviceRepr, TR: DeviceRepr>(
    x: &(impl DeviceMatrixChunkImpl<T0> + ?Sized),
    y: &(impl DeviceMatrixChunkImpl<T1> + ?Sized),
    result: &mut (impl DeviceMatrixChunkMutImpl<TR> + ?Sized),
    stream: &CudaStream,
) -> CudaResult<()>
where
    (T0, T1, TR): BinaryOp<T0, T1, TR>,
{
    binary_op(BinaryOpType::Mul, x, y, result, stream)
}

pub fn mul_into_x<T0: DeviceRepr, T1: DeviceRepr>(
    x: &mut (impl DeviceMatrixChunkMutImpl<T0> + ?Sized),
    y: &(impl DeviceMatrixChunkImpl<T1> + ?Sized),
    stream: &CudaStream,
) -> CudaResult<()>
where
    (T0, T1, T0): BinaryOp<T0, T1, T0>,
{
    binary_op_into_x(BinaryOpType::Mul, x, y, stream)
}

pub fn mul_into_y<T0: DeviceRepr, T1: DeviceRepr>(
    x: &(impl DeviceMatrixChunkImpl<T0> + ?Sized),
    y: &mut (impl DeviceMatrixChunkMutImpl<T1> + ?Sized),
    stream: &CudaStream,
) -> CudaResult<()>
where
    (T0, T1, T1): BinaryOp<T0, T1, T1>,
{
    binary_op_into_y(BinaryOpType::Mul, x, y, stream)
}

pub fn sub<T0: DeviceRepr, T1: DeviceRepr, TR: DeviceRepr>(
    x: &(impl DeviceMatrixChunkImpl<T0> + ?Sized),
    y: &(impl DeviceMatrixChunkImpl<T1> + ?Sized),
    result: &mut (impl DeviceMatrixChunkMutImpl<TR> + ?Sized),
    stream: &CudaStream,
) -> CudaResult<()>
where
    (T0, T1, TR): BinaryOp<T0, T1, TR>,
{
    binary_op(BinaryOpType::Sub, x, y, result, stream)
}

pub fn sub_into_x<T0: DeviceRepr, T1: DeviceRepr>(
    x: &mut (impl DeviceMatrixChunkMutImpl<T0> + ?Sized),
    y: &(impl DeviceMatrixChunkImpl<T1> + ?Sized),
    stream: &CudaStream,
) -> CudaResult<()>
where
    (T0, T1, T0): BinaryOp<T0, T1, T0>,
{
    binary_op_into_x(BinaryOpType::Sub, x, y, stream)
}

pub fn sub_into_y<T0: DeviceRepr, T1: DeviceRepr>(
    x: &(impl DeviceMatrixChunkImpl<T0> + ?Sized),
    y: &mut (impl DeviceMatrixChunkMutImpl<T1> + ?Sized),
    stream: &CudaStream,
) -> CudaResult<()>
where
    (T0, T1, T1): BinaryOp<T0, T1, T1>,
{
    binary_op_into_y(BinaryOpType::Sub, x, y, stream)
}

pub enum TernaryOpType {
    MulAdd,
    MulSub,
}

type TernaryKernel<T0, T1, T2, TR> = KernelFourArgs<
    PtrAndStrideWrappingMatrix<<T0 as DeviceRepr>::Type>,
    PtrAndStrideWrappingMatrix<<T1 as DeviceRepr>::Type>,
    PtrAndStrideWrappingMatrix<<T2 as DeviceRepr>::Type>,
    MutPtrAndStrideWrappingMatrix<<TR as DeviceRepr>::Type>,
>;

pub trait TernaryOp<
    T0: Sized + DeviceRepr,
    T1: Sized + DeviceRepr,
    T2: Sized + DeviceRepr,
    TR: Sized + DeviceRepr,
>
{
    fn get_kernel(op_type: TernaryOpType) -> TernaryKernel<T0, T1, T2, TR>;

    fn launch_op(
        op_type: TernaryOpType,
        x: PtrAndStrideWrappingMatrix<T0::Type>,
        y: PtrAndStrideWrappingMatrix<T1::Type>,
        z: PtrAndStrideWrappingMatrix<T2::Type>,
        result: MutPtrAndStrideWrappingMatrix<TR::Type>,
        stream: &CudaStream,
    ) -> CudaResult<()> {
        let kernel = Self::get_kernel(op_type);
        let (grid_dim, block_dim) = get_launch_dims(result.rows, result.cols);
        let args = (&x, &y, &z, &result);
        unsafe { kernel.launch(grid_dim, block_dim, args, 0, stream) }
    }

    fn launch(
        op_type: TernaryOpType,
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
            op_type,
            PtrAndStrideWrappingMatrix::new(x),
            PtrAndStrideWrappingMatrix::new(y),
            PtrAndStrideWrappingMatrix::new(z),
            MutPtrAndStrideWrappingMatrix::new(result),
            stream,
        )
    }

    fn launch_into_x(
        op_type: TernaryOpType,
        x: &mut (impl DeviceMatrixChunkMutImpl<T0> + ?Sized),
        y: &(impl DeviceMatrixChunkImpl<T1> + ?Sized),
        z: &(impl DeviceMatrixChunkImpl<T2> + ?Sized),
        stream: &CudaStream,
    ) -> CudaResult<()>
    where
        (T0, T1, T2, T0): TernaryOp<T0, T1, T2, T0>,
    {
        <(T0, T1, T2, T0)>::launch_op(
            op_type,
            PtrAndStrideWrappingMatrix::new(x),
            PtrAndStrideWrappingMatrix::new(y),
            PtrAndStrideWrappingMatrix::new(z),
            MutPtrAndStrideWrappingMatrix::new(x),
            stream,
        )
    }

    fn launch_into_y(
        op_type: TernaryOpType,
        x: &(impl DeviceMatrixChunkImpl<T0> + ?Sized),
        y: &mut (impl DeviceMatrixChunkMutImpl<T1> + ?Sized),
        z: &(impl DeviceMatrixChunkImpl<T2> + ?Sized),
        stream: &CudaStream,
    ) -> CudaResult<()>
    where
        (T0, T1, T2, T1): TernaryOp<T0, T1, T2, T1>,
    {
        <(T0, T1, T2, T1)>::launch_op(
            op_type,
            PtrAndStrideWrappingMatrix::new(x),
            PtrAndStrideWrappingMatrix::new(y),
            PtrAndStrideWrappingMatrix::new(z),
            MutPtrAndStrideWrappingMatrix::new(y),
            stream,
        )
    }

    fn launch_into_z(
        op_type: TernaryOpType,
        x: &(impl DeviceMatrixChunkImpl<T0> + ?Sized),
        y: &(impl DeviceMatrixChunkImpl<T1> + ?Sized),
        z: &mut (impl DeviceMatrixChunkMutImpl<T2> + ?Sized),
        stream: &CudaStream,
    ) -> CudaResult<()>
    where
        (T0, T1, T2, T2): TernaryOp<T0, T1, T2, T2>,
    {
        <(T0, T1, T2, T2)>::launch_op(
            op_type,
            PtrAndStrideWrappingMatrix::new(x),
            PtrAndStrideWrappingMatrix::new(y),
            PtrAndStrideWrappingMatrix::new(z),
            MutPtrAndStrideWrappingMatrix::new(z),
            stream,
        )
    }
}

impl TernaryOp<BaseField, BaseField, BaseField, BaseField>
    for (BaseField, BaseField, BaseField, BaseField)
{
    fn get_kernel(
        op_type: TernaryOpType,
    ) -> TernaryKernel<BaseField, BaseField, BaseField, BaseField> {
        match op_type {
            TernaryOpType::MulAdd => mul_add_bf_bf_bf_kernel,
            TernaryOpType::MulSub => mul_sub_bf_bf_bf_kernel,
        }
    }
}

impl TernaryOp<BaseField, BaseField, VectorizedExtensionField, VectorizedExtensionField>
    for (
        BaseField,
        BaseField,
        VectorizedExtensionField,
        VectorizedExtensionField,
    )
{
    fn get_kernel(
        op_type: TernaryOpType,
    ) -> TernaryKernel<BaseField, BaseField, VectorizedExtensionField, VectorizedExtensionField>
    {
        match op_type {
            TernaryOpType::MulAdd => mul_add_bf_bf_ef_kernel,
            TernaryOpType::MulSub => mul_sub_bf_bf_ef_kernel,
        }
    }
}

impl TernaryOp<BaseField, VectorizedExtensionField, BaseField, VectorizedExtensionField>
    for (
        BaseField,
        VectorizedExtensionField,
        BaseField,
        VectorizedExtensionField,
    )
{
    fn get_kernel(
        op_type: TernaryOpType,
    ) -> TernaryKernel<BaseField, VectorizedExtensionField, BaseField, VectorizedExtensionField>
    {
        match op_type {
            TernaryOpType::MulAdd => mul_add_bf_ef_bf_kernel,
            TernaryOpType::MulSub => mul_sub_bf_ef_bf_kernel,
        }
    }
}

impl
    TernaryOp<
        BaseField,
        VectorizedExtensionField,
        VectorizedExtensionField,
        VectorizedExtensionField,
    >
    for (
        BaseField,
        VectorizedExtensionField,
        VectorizedExtensionField,
        VectorizedExtensionField,
    )
{
    fn get_kernel(
        op_type: TernaryOpType,
    ) -> TernaryKernel<
        BaseField,
        VectorizedExtensionField,
        VectorizedExtensionField,
        VectorizedExtensionField,
    > {
        match op_type {
            TernaryOpType::MulAdd => mul_add_bf_ef_ef_kernel,
            TernaryOpType::MulSub => mul_sub_bf_ef_ef_kernel,
        }
    }
}

impl TernaryOp<VectorizedExtensionField, BaseField, BaseField, VectorizedExtensionField>
    for (
        VectorizedExtensionField,
        BaseField,
        BaseField,
        VectorizedExtensionField,
    )
{
    fn get_kernel(
        op_type: TernaryOpType,
    ) -> TernaryKernel<VectorizedExtensionField, BaseField, BaseField, VectorizedExtensionField>
    {
        match op_type {
            TernaryOpType::MulAdd => mul_add_ef_bf_bf_kernel,
            TernaryOpType::MulSub => mul_sub_ef_bf_bf_kernel,
        }
    }
}

impl
    TernaryOp<
        VectorizedExtensionField,
        BaseField,
        VectorizedExtensionField,
        VectorizedExtensionField,
    >
    for (
        VectorizedExtensionField,
        BaseField,
        VectorizedExtensionField,
        VectorizedExtensionField,
    )
{
    fn get_kernel(
        op_type: TernaryOpType,
    ) -> TernaryKernel<
        VectorizedExtensionField,
        BaseField,
        VectorizedExtensionField,
        VectorizedExtensionField,
    > {
        match op_type {
            TernaryOpType::MulAdd => mul_add_ef_bf_ef_kernel,
            TernaryOpType::MulSub => mul_sub_ef_bf_ef_kernel,
        }
    }
}

impl
    TernaryOp<
        VectorizedExtensionField,
        VectorizedExtensionField,
        BaseField,
        VectorizedExtensionField,
    >
    for (
        VectorizedExtensionField,
        VectorizedExtensionField,
        BaseField,
        VectorizedExtensionField,
    )
{
    fn get_kernel(
        op_type: TernaryOpType,
    ) -> TernaryKernel<
        VectorizedExtensionField,
        VectorizedExtensionField,
        BaseField,
        VectorizedExtensionField,
    > {
        match op_type {
            TernaryOpType::MulAdd => mul_add_ef_ef_bf_kernel,
            TernaryOpType::MulSub => mul_sub_ef_ef_bf_kernel,
        }
    }
}

impl
    TernaryOp<
        VectorizedExtensionField,
        VectorizedExtensionField,
        VectorizedExtensionField,
        VectorizedExtensionField,
    >
    for (
        VectorizedExtensionField,
        VectorizedExtensionField,
        VectorizedExtensionField,
        VectorizedExtensionField,
    )
{
    fn get_kernel(
        op_type: TernaryOpType,
    ) -> TernaryKernel<
        VectorizedExtensionField,
        VectorizedExtensionField,
        VectorizedExtensionField,
        VectorizedExtensionField,
    > {
        match op_type {
            TernaryOpType::MulAdd => mul_add_ef_ef_ef_kernel,
            TernaryOpType::MulSub => mul_sub_ef_ef_ef_kernel,
        }
    }
}

fn ternary_op<T0: DeviceRepr, T1: DeviceRepr, T2: DeviceRepr, TR: DeviceRepr>(
    op_type: TernaryOpType,
    x: &(impl DeviceMatrixChunkImpl<T0> + ?Sized),
    y: &(impl DeviceMatrixChunkImpl<T1> + ?Sized),
    z: &(impl DeviceMatrixChunkImpl<T2> + ?Sized),
    result: &mut (impl DeviceMatrixChunkMutImpl<TR> + ?Sized),
    stream: &CudaStream,
) -> CudaResult<()>
where
    (T0, T1, T2, TR): TernaryOp<T0, T1, T2, TR>,
{
    <(T0, T1, T2, TR)>::launch(op_type, x, y, z, result, stream)
}

fn ternary_op_into_x<T0: DeviceRepr, T1: DeviceRepr, T2: DeviceRepr>(
    op_type: TernaryOpType,
    x: &mut (impl DeviceMatrixChunkMutImpl<T0> + ?Sized),
    y: &(impl DeviceMatrixChunkImpl<T1> + ?Sized),
    z: &(impl DeviceMatrixChunkImpl<T2> + ?Sized),
    stream: &CudaStream,
) -> CudaResult<()>
where
    (T0, T1, T2, T0): TernaryOp<T0, T1, T2, T0>,
{
    <(T0, T1, T2, T0)>::launch_into_x(op_type, x, y, z, stream)
}

fn ternary_op_into_y<T0: DeviceRepr, T1: DeviceRepr, T2: DeviceRepr>(
    op_type: TernaryOpType,
    x: &(impl DeviceMatrixChunkImpl<T0> + ?Sized),
    y: &mut (impl DeviceMatrixChunkMutImpl<T1> + ?Sized),
    z: &(impl DeviceMatrixChunkImpl<T2> + ?Sized),
    stream: &CudaStream,
) -> CudaResult<()>
where
    (T0, T1, T2, T1): TernaryOp<T0, T1, T2, T1>,
{
    <(T0, T1, T2, T1)>::launch_into_y(op_type, x, y, z, stream)
}

fn ternary_op_into_z<T0: DeviceRepr, T1: DeviceRepr, T2: DeviceRepr>(
    op_type: TernaryOpType,
    x: &(impl DeviceMatrixChunkImpl<T0> + ?Sized),
    y: &(impl DeviceMatrixChunkImpl<T1> + ?Sized),
    z: &mut (impl DeviceMatrixChunkMutImpl<T2> + ?Sized),
    stream: &CudaStream,
) -> CudaResult<()>
where
    (T0, T1, T2, T2): TernaryOp<T0, T1, T2, T2>,
{
    <(T0, T1, T2, T2)>::launch_into_z(op_type, x, y, z, stream)
}

pub fn mul_add<T0: DeviceRepr, T1: DeviceRepr, T2: DeviceRepr, TR: DeviceRepr>(
    x: &(impl DeviceMatrixChunkImpl<T0> + ?Sized),
    y: &(impl DeviceMatrixChunkImpl<T1> + ?Sized),
    z: &(impl DeviceMatrixChunkImpl<T2> + ?Sized),
    result: &mut (impl DeviceMatrixChunkMutImpl<TR> + ?Sized),
    stream: &CudaStream,
) -> CudaResult<()>
where
    (T0, T1, T2, TR): TernaryOp<T0, T1, T2, TR>,
{
    ternary_op(TernaryOpType::MulAdd, x, y, z, result, stream)
}

pub fn mul_add_into_x<T0: DeviceRepr, T1: DeviceRepr, T2: DeviceRepr>(
    x: &mut (impl DeviceMatrixChunkMutImpl<T0> + ?Sized),
    y: &(impl DeviceMatrixChunkImpl<T1> + ?Sized),
    z: &(impl DeviceMatrixChunkImpl<T2> + ?Sized),
    stream: &CudaStream,
) -> CudaResult<()>
where
    (T0, T1, T2, T0): TernaryOp<T0, T1, T2, T0>,
{
    ternary_op_into_x(TernaryOpType::MulAdd, x, y, z, stream)
}

pub fn mul_add_into_y<T0: DeviceRepr, T1: DeviceRepr, T2: DeviceRepr>(
    x: &(impl DeviceMatrixChunkImpl<T0> + ?Sized),
    y: &mut (impl DeviceMatrixChunkMutImpl<T1> + ?Sized),
    z: &(impl DeviceMatrixChunkImpl<T2> + ?Sized),
    stream: &CudaStream,
) -> CudaResult<()>
where
    (T0, T1, T2, T1): TernaryOp<T0, T1, T2, T1>,
{
    ternary_op_into_y(TernaryOpType::MulAdd, x, y, z, stream)
}

pub fn mul_add_into_z<T0: DeviceRepr, T1: DeviceRepr, T2: DeviceRepr>(
    x: &(impl DeviceMatrixChunkImpl<T0> + ?Sized),
    y: &(impl DeviceMatrixChunkImpl<T1> + ?Sized),
    z: &mut (impl DeviceMatrixChunkMutImpl<T2> + ?Sized),
    stream: &CudaStream,
) -> CudaResult<()>
where
    (T0, T1, T2, T2): TernaryOp<T0, T1, T2, T2>,
{
    ternary_op_into_z(TernaryOpType::MulAdd, x, y, z, stream)
}

pub fn mul_sub<T0: DeviceRepr, T1: DeviceRepr, T2: DeviceRepr, TR: DeviceRepr>(
    x: &(impl DeviceMatrixChunkImpl<T0> + ?Sized),
    y: &(impl DeviceMatrixChunkImpl<T1> + ?Sized),
    z: &(impl DeviceMatrixChunkImpl<T2> + ?Sized),
    result: &mut (impl DeviceMatrixChunkMutImpl<TR> + ?Sized),
    stream: &CudaStream,
) -> CudaResult<()>
where
    (T0, T1, T2, TR): TernaryOp<T0, T1, T2, TR>,
{
    ternary_op(TernaryOpType::MulSub, x, y, z, result, stream)
}

pub fn mul_sub_into_x<T0: DeviceRepr, T1: DeviceRepr, T2: DeviceRepr>(
    x: &mut (impl DeviceMatrixChunkMutImpl<T0> + ?Sized),
    y: &(impl DeviceMatrixChunkImpl<T1> + ?Sized),
    z: &(impl DeviceMatrixChunkImpl<T2> + ?Sized),
    stream: &CudaStream,
) -> CudaResult<()>
where
    (T0, T1, T2, T0): TernaryOp<T0, T1, T2, T0>,
{
    ternary_op_into_x(TernaryOpType::MulSub, x, y, z, stream)
}

pub fn mul_sub_into_y<T0: DeviceRepr, T1: DeviceRepr, T2: DeviceRepr>(
    x: &(impl DeviceMatrixChunkImpl<T0> + ?Sized),
    y: &mut (impl DeviceMatrixChunkMutImpl<T1> + ?Sized),
    z: &(impl DeviceMatrixChunkImpl<T2> + ?Sized),
    stream: &CudaStream,
) -> CudaResult<()>
where
    (T0, T1, T2, T1): TernaryOp<T0, T1, T2, T1>,
{
    ternary_op_into_y(TernaryOpType::MulSub, x, y, z, stream)
}

pub fn mul_sub_into_z<T0: DeviceRepr, T1: DeviceRepr, T2: DeviceRepr>(
    x: &(impl DeviceMatrixChunkImpl<T0> + ?Sized),
    y: &(impl DeviceMatrixChunkImpl<T1> + ?Sized),
    z: &mut (impl DeviceMatrixChunkMutImpl<T2> + ?Sized),
    stream: &CudaStream,
) -> CudaResult<()>
where
    (T0, T1, T2, T2): TernaryOp<T0, T1, T2, T2>,
{
    ternary_op_into_z(TernaryOpType::MulSub, x, y, z, stream)
}

#[cfg(test)]
mod tests {
    use std::ops::{Add, Mul, Sub};

    use boojum::field::goldilocks::GoldilocksField;
    use boojum::field::{Field, PrimeField, U64Representable};
    use itertools::Itertools;

    use cudart::memory::{memory_copy_async, DeviceAllocation};
    use cudart::result::CudaResult;
    use cudart::slice::DeviceSlice;
    use cudart::stream::CudaStream;

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
             stream: &CudaStream| (super::add_into_y(x, y, stream));
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
             stream: &CudaStream| (super::mul_into_y(y, x, stream));
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
