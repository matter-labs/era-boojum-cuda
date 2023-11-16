use boojum::field::goldilocks::{GoldilocksExt2, GoldilocksField};

use cudart::execution::{KernelFourArgs, KernelLaunch};
use cudart::result::CudaResult;
use cudart::stream::CudaStream;

use crate::device_structures::{
    DeviceMatrixChunkImpl, DeviceMatrixChunkMutImpl, DeviceRepr, ExtensionFieldDeviceType,
    MutPtrAndStride, PtrAndStride, VectorizedExtensionFieldDeviceType,
};
use crate::utils::{get_grid_block_dims_for_threads_count, WARP_SIZE};

pub type ExtensionField = boojum::field::ExtensionField<GoldilocksField, 2, GoldilocksExt2>;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct VectorizedExtensionField([GoldilocksField; 2]);

extern "C" {
    fn vectorized_to_tuples_kernel(
        src: PtrAndStride<VectorizedExtensionFieldDeviceType>,
        dst: MutPtrAndStride<ExtensionFieldDeviceType>,
        rows: u32,
        cols: u32,
    );

    fn tuples_to_vectorized_kernel(
        src: PtrAndStride<ExtensionFieldDeviceType>,
        dst: MutPtrAndStride<VectorizedExtensionFieldDeviceType>,
        rows: u32,
        cols: u32,
    );
}

pub trait ConvertTarget: DeviceRepr {
    type Target: DeviceRepr;
}

type ConvertKernel<T> = KernelFourArgs<
    PtrAndStride<<T as DeviceRepr>::Type>,
    MutPtrAndStride<<<T as ConvertTarget>::Target as DeviceRepr>::Type>,
    u32,
    u32,
>;

pub trait Convert: ConvertTarget {
    fn get_kernel() -> ConvertKernel<Self>;

    fn launch_kernel<S, D>(src: &S, dst: &mut D, stream: &CudaStream) -> CudaResult<()>
    where
        S: DeviceMatrixChunkImpl<Self> + ?Sized,
        D: DeviceMatrixChunkMutImpl<Self::Target> + ?Sized,
    {
        assert_eq!(src.rows(), dst.rows());
        assert!(src.rows() <= u32::MAX as usize);
        let rows = src.rows() as u32;
        assert_eq!(src.cols(), dst.cols());
        assert!(src.cols() <= u32::MAX as usize);
        let cols = src.cols() as u32;
        let src = src.as_ptr_and_stride();
        let dst = dst.as_mut_ptr_and_stride();
        let (grid_dim, block_dim) =
            get_grid_block_dims_for_threads_count(WARP_SIZE * 4, rows * cols);
        let args = (&src, &dst, &rows, &cols);
        let kernel = Self::get_kernel();
        unsafe { kernel.launch(grid_dim, block_dim, args, 0, stream) }
    }
}

impl ConvertTarget for ExtensionField {
    type Target = VectorizedExtensionField;
}

impl Convert for ExtensionField {
    fn get_kernel() -> ConvertKernel<Self> {
        tuples_to_vectorized_kernel
    }
}

impl ConvertTarget for VectorizedExtensionField {
    type Target = ExtensionField;
}

impl Convert for VectorizedExtensionField {
    fn get_kernel() -> ConvertKernel<Self> {
        vectorized_to_tuples_kernel
    }
}

pub fn convert<T, S, D>(src: &S, dst: &mut D, stream: &CudaStream) -> CudaResult<()>
where
    T: Convert,
    S: DeviceMatrixChunkImpl<T> + ?Sized,
    D: DeviceMatrixChunkMutImpl<T::Target> + ?Sized,
{
    T::launch_kernel(src, dst, stream)
}

#[cfg(test)]
pub(crate) mod test_helpers {
    use std::iter::{Map, Zip};
    use std::slice;
    use std::slice::Iter;

    use boojum::field::goldilocks::GoldilocksField;

    use super::*;

    type VectorizedExtensionFieldIteratorInner<'a> = Map<
        Zip<Iter<'a, GoldilocksField>, Iter<'a, GoldilocksField>>,
        fn((&GoldilocksField, &GoldilocksField)) -> ExtensionField,
    >;

    pub struct VectorizedExtensionFieldIterator<'a>(VectorizedExtensionFieldIteratorInner<'a>);

    impl<'a> VectorizedExtensionFieldIterator<'a> {
        pub fn new(slice: &'a [VectorizedExtensionField]) -> Self {
            let (c0, c1) = (unsafe {
                slice::from_raw_parts(slice.as_ptr() as *const GoldilocksField, slice.len() * 2)
            })
            .split_at(slice.len());
            let iter: VectorizedExtensionFieldIteratorInner<'a> =
                c0.iter().zip(c1.iter()).map(Self::map_fn);
            Self(iter)
        }

        fn map_fn((c0, c1): (&GoldilocksField, &GoldilocksField)) -> ExtensionField {
            ExtensionField::from_coeff_in_base([*c0, *c1])
        }
    }

    impl<'a> Iterator for VectorizedExtensionFieldIterator<'a> {
        type Item = ExtensionField;

        fn next(&mut self) -> Option<Self::Item> {
            self.0.next()
        }
    }

    pub trait ExtensionFieldTestSuper: Convert {
        type Iterator<'a>: Iterator<Item = ExtensionField>;
    }

    pub trait ExtensionFieldTest: ExtensionFieldTestSuper {
        fn get_iterator(slice: &[Self]) -> Self::Iterator<'_>;
    }

    impl ExtensionFieldTestSuper for ExtensionField {
        type Iterator<'a> = Map<Iter<'a, ExtensionField>, fn(&ExtensionField) -> ExtensionField>;
    }

    impl ExtensionFieldTest for ExtensionField {
        fn get_iterator(slice: &[Self]) -> Self::Iterator<'_> {
            slice.iter().map(ExtensionField::clone)
        }
    }

    impl ExtensionFieldTestSuper for VectorizedExtensionField {
        type Iterator<'a> = VectorizedExtensionFieldIterator<'a>;
    }

    impl ExtensionFieldTest for VectorizedExtensionField {
        fn get_iterator(slice: &[Self]) -> Self::Iterator<'_> {
            VectorizedExtensionFieldIterator::new(slice)
        }
    }

    pub fn transmute_gf_vec<T: ExtensionFieldTest>(vec: Vec<GoldilocksField>) -> Vec<T> {
        assert_eq!(vec.len() % 2, 0);
        let (ptr, len, cap) = vec.into_raw_parts();
        unsafe { Vec::from_raw_parts(ptr as *mut T, len / 2, cap / 2) }
    }
}

#[cfg(test)]
mod tests {
    use std::{mem, vec};

    use boojum::field::goldilocks::GoldilocksField;
    use boojum::field::Field;
    use itertools::Itertools;
    use rand::distributions::Uniform;
    use rand::prelude::*;

    use cudart::memory::{memory_copy_async, DeviceAllocation};
    use cudart::stream::CudaStream;

    use crate::device_structures::{DeviceMatrix, DeviceMatrixMut};
    use crate::extension_field::{ExtensionField, VectorizedExtensionField};

    use super::test_helpers::*;

    #[test]
    fn extension_field_size() {
        assert_eq!(mem::size_of::<ExtensionField>(), 16);
    }

    #[test]
    fn extension_field_align() {
        assert_eq!(mem::align_of::<ExtensionField>(), 8);
    }

    #[test]
    fn vectorized_extension_field_size() {
        assert_eq!(mem::size_of::<VectorizedExtensionField>(), 16);
    }

    #[test]
    fn vectorized_extension_field_align() {
        assert_eq!(mem::align_of::<VectorizedExtensionField>(), 8);
    }

    fn test_conversion<T>()
    where
        T: ExtensionFieldTest,
        T::Target: ExtensionFieldTest,
    {
        const ROWS: usize = 1 << 8;
        const COLS: usize = 1 << 10;
        const N: usize = ROWS * COLS;
        let h_src = transmute_gf_vec(
            Uniform::new(0, GoldilocksField::ORDER)
                .sample_iter(&mut thread_rng())
                .take(N * 2)
                .map(GoldilocksField)
                .collect_vec(),
        );
        let mut h_dst = transmute_gf_vec(vec![GoldilocksField::ZERO; N * 2]);
        let mut d_src = DeviceAllocation::alloc(N).unwrap();
        let mut d_dst = DeviceAllocation::alloc(N).unwrap();
        let stream = CudaStream::default();
        memory_copy_async(&mut d_src, &h_src, &stream).unwrap();
        let src = DeviceMatrix::new(&d_src, ROWS);
        let mut dst = DeviceMatrixMut::new(&mut d_dst, ROWS);
        super::convert(&src, &mut dst, &stream).unwrap();
        memory_copy_async(&mut h_dst, &d_dst, &stream).unwrap();
        stream.synchronize().unwrap();
        h_src
            .chunks(ROWS)
            .zip(h_dst.chunks(ROWS))
            .for_each(|(src, dst)| {
                T::get_iterator(src)
                    .zip(T::Target::get_iterator(dst))
                    .for_each(|(a, b)| assert_eq!(a, b));
            });
    }

    #[test]
    fn vectorized_to_tuples() {
        test_conversion::<VectorizedExtensionField>();
    }

    #[test]
    fn tuples_to_vectorized() {
        test_conversion::<ExtensionField>();
    }
}
