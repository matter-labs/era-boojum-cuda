use std::ops;
use std::ops::{Index, IndexMut};

use crate::slice::{CudaSlice, CudaSliceMut, DeviceSlice, DeviceVariable};

impl<T, I> Index<I> for DeviceSlice<T>
where
    I: DeviceSliceIndex<T>,
{
    type Output = I::Output;

    #[inline]
    fn index(&self, index: I) -> &I::Output {
        index.index(self)
    }
}

impl<T, I> IndexMut<I> for DeviceSlice<T>
where
    I: DeviceSliceIndex<T>,
{
    #[inline]
    fn index_mut(&mut self, index: I) -> &mut I::Output {
        index.index_mut(self)
    }
}

pub trait DeviceSliceIndex<T> {
    type Output: ?Sized;

    fn index(self, slice: &DeviceSlice<T>) -> &Self::Output;

    fn index_mut(self, slice: &mut DeviceSlice<T>) -> &mut Self::Output;
}

impl<T> DeviceSliceIndex<T> for usize {
    type Output = DeviceVariable<T>;

    fn index(self, slice: &DeviceSlice<T>) -> &Self::Output {
        unsafe { DeviceVariable::from_ref(slice.as_slice().index(self)) }
    }

    fn index_mut(self, slice: &mut DeviceSlice<T>) -> &mut Self::Output {
        unsafe { DeviceVariable::from_mut(slice.as_mut_slice().index_mut(self)) }
    }
}

trait DeviceSliceToSliceIndex {}

impl<T, I> DeviceSliceIndex<T> for I
where
    I: DeviceSliceToSliceIndex,
    [T]: Index<I, Output = [T]> + IndexMut<I, Output = [T]>,
{
    type Output = DeviceSlice<T>;

    fn index(self, slice: &DeviceSlice<T>) -> &Self::Output {
        unsafe { DeviceSlice::from_slice(slice.as_slice().index(self)) }
    }

    fn index_mut(self, slice: &mut DeviceSlice<T>) -> &mut Self::Output {
        unsafe { DeviceSlice::from_mut_slice(slice.as_mut_slice().index_mut(self)) }
    }
}

impl DeviceSliceToSliceIndex for ops::RangeFull {}

impl DeviceSliceToSliceIndex for ops::Range<usize> {}

impl DeviceSliceToSliceIndex for ops::RangeFrom<usize> {}

impl DeviceSliceToSliceIndex for ops::RangeTo<usize> {}

impl DeviceSliceToSliceIndex for ops::RangeInclusive<usize> {}

impl DeviceSliceToSliceIndex for ops::RangeToInclusive<usize> {}

impl DeviceSliceToSliceIndex for (ops::Bound<usize>, ops::Bound<usize>) {}
