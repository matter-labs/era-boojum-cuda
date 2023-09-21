#![allow(incomplete_features)]
#![feature(allocator_api)]
#![feature(generic_const_exprs)]
#![feature(inline_const)]
#![feature(iter_array_chunks)]
#![feature(macro_metavar_expr)]
#![feature(specialization)]
#![feature(vec_into_raw_parts)]

pub mod barycentric;
pub mod blake2s;
pub mod context;
pub mod device_structures;
pub mod extension_field;
pub mod gates;
pub mod ntt;
pub mod ops_complex;
pub mod ops_cub;
pub mod ops_simple;
pub mod poseidon;
mod utils;

pub type BaseField = boojum::field::goldilocks::GoldilocksField;

#[cfg(test)]
mod tests_helpers;
