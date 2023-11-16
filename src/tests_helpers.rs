use std::iter::{ArrayChunks, Map};

use boojum::field::goldilocks::GoldilocksField;
use rand::distributions::Uniform;
use rand::prelude::*;

use crate::extension_field::ExtensionField;

pub trait RandomIterator: Sized {
    type Iterator: Iterator<Item = Self>;

    fn get_random_iterator() -> Self::Iterator;
}

pub struct GoldilocksFieldRandomIterator {
    uniform: Uniform<u64>,
    rng: ThreadRng,
}

impl GoldilocksFieldRandomIterator {
    fn new() -> Self {
        Self {
            uniform: Uniform::new(0, GoldilocksField::ORDER),
            rng: thread_rng(),
        }
    }
}

impl Iterator for GoldilocksFieldRandomIterator {
    type Item = GoldilocksField;

    fn next(&mut self) -> Option<Self::Item> {
        Some(GoldilocksField(self.uniform.sample(&mut self.rng)))
    }
}

impl RandomIterator for GoldilocksField {
    type Iterator = GoldilocksFieldRandomIterator;

    fn get_random_iterator() -> Self::Iterator {
        GoldilocksFieldRandomIterator::new()
    }
}

type ExtensionFieldRandomIteratorInner =
    Map<ArrayChunks<GoldilocksFieldRandomIterator, 2>, fn([GoldilocksField; 2]) -> ExtensionField>;

pub struct ExtensionFieldRandomIterator {
    iterator: ExtensionFieldRandomIteratorInner,
}

impl ExtensionFieldRandomIterator {
    fn new() -> Self {
        Self {
            iterator: GoldilocksFieldRandomIterator::new()
                .array_chunks()
                .map(ExtensionField::from_coeff_in_base),
        }
    }
}

impl Iterator for ExtensionFieldRandomIterator {
    type Item = ExtensionField;

    fn next(&mut self) -> Option<Self::Item> {
        self.iterator.next()
    }
}

impl RandomIterator for ExtensionField {
    type Iterator = ExtensionFieldRandomIterator;

    fn get_random_iterator() -> Self::Iterator {
        ExtensionFieldRandomIterator::new()
    }
}

mod tests {
    use super::*;

    #[test]
    fn generate_bf() {
        let _ = GoldilocksField::get_random_iterator().next().unwrap();
    }

    #[test]
    fn generate_ef() {
        let _ = ExtensionField::get_random_iterator().next().unwrap();
    }
}
