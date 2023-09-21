use std::iter::{FusedIterator, TrustedLen, TrustedRandomAccess, TrustedRandomAccessNoCoerce};
use std::slice;

use crate::slice::cuda_slice::{CudaSlice, CudaSliceMut};
use crate::slice::device_slice::DeviceSlice;

#[repr(transparent)]
pub struct Chunks<'a, T>(slice::Chunks<'a, T>);

impl<'a, T: 'a> Chunks<'a, T> {
    #[inline]
    pub(super) fn new(slice: &'a DeviceSlice<T>, size: usize) -> Self {
        unsafe { Self(slice.as_slice().chunks(size)) }
    }

    #[inline]
    fn map_option(option: Option<&'a [T]>) -> Option<&'a DeviceSlice<T>> {
        unsafe { option.map(|s| DeviceSlice::from_slice(s)) }
    }
}

impl<T> Clone for Chunks<'_, T> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<'a, T> Iterator for Chunks<'a, T> {
    type Item = &'a DeviceSlice<T>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        Self::map_option(self.0.next())
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }

    #[inline]
    fn count(self) -> usize {
        self.0.len()
    }

    #[inline]
    fn last(self) -> Option<Self::Item> {
        Self::map_option(self.0.last())
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        Self::map_option(self.0.nth(n))
    }

    unsafe fn __iterator_get_unchecked(&mut self, idx: usize) -> Self::Item {
        DeviceSlice::from_slice(self.0.__iterator_get_unchecked(idx))
    }
}

impl<'a, T> DoubleEndedIterator for Chunks<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        Self::map_option(self.0.next_back())
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        Self::map_option(self.0.nth_back(n))
    }
}

impl<T> ExactSizeIterator for Chunks<'_, T> {}

unsafe impl<T> TrustedLen for Chunks<'_, T> {}

impl<T> FusedIterator for Chunks<'_, T> {}

#[doc(hidden)]
unsafe impl<'a, T> TrustedRandomAccess for Chunks<'a, T> {}

#[doc(hidden)]
unsafe impl<'a, T> TrustedRandomAccessNoCoerce for Chunks<'a, T> {
    const MAY_HAVE_SIDE_EFFECT: bool = false;
}

#[repr(transparent)]
pub struct ChunksMut<'a, T>(slice::ChunksMut<'a, T>);

impl<'a, T: 'a> ChunksMut<'a, T> {
    #[inline]
    pub(super) fn new(slice: &'a mut DeviceSlice<T>, size: usize) -> Self {
        unsafe { Self(slice.as_mut_slice().chunks_mut(size)) }
    }

    #[inline]
    fn map_option(option: Option<&'a mut [T]>) -> Option<&'a mut DeviceSlice<T>> {
        unsafe { option.map(|s| DeviceSlice::from_mut_slice(s)) }
    }
}

impl<'a, T> Iterator for ChunksMut<'a, T> {
    type Item = &'a mut DeviceSlice<T>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        Self::map_option(self.0.next())
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }

    #[inline]
    fn count(self) -> usize {
        self.len()
    }

    #[inline]
    fn last(self) -> Option<Self::Item> {
        Self::map_option(self.0.last())
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        Self::map_option(self.0.nth(n))
    }

    unsafe fn __iterator_get_unchecked(&mut self, idx: usize) -> Self::Item {
        DeviceSlice::from_mut_slice(self.0.__iterator_get_unchecked(idx))
    }
}

impl<'a, T> DoubleEndedIterator for ChunksMut<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        Self::map_option(self.0.next_back())
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        Self::map_option(self.0.nth_back(n))
    }
}

impl<T> ExactSizeIterator for ChunksMut<'_, T> {}

unsafe impl<T> TrustedLen for ChunksMut<'_, T> {}

impl<T> FusedIterator for ChunksMut<'_, T> {}

#[doc(hidden)]
unsafe impl<'a, T> TrustedRandomAccess for ChunksMut<'a, T> {}

#[doc(hidden)]
unsafe impl<'a, T> TrustedRandomAccessNoCoerce for ChunksMut<'a, T> {
    const MAY_HAVE_SIDE_EFFECT: bool = false;
}

unsafe impl<T> Send for ChunksMut<'_, T> where T: Send {}

unsafe impl<T> Sync for ChunksMut<'_, T> where T: Sync {}
