#![feature(min_specialization)]
#![feature(ptr_metadata)]
#![feature(trusted_len)]
#![feature(trusted_random_access)]

extern crate core;

pub mod device;
pub mod error;
pub mod event;
pub mod execution;
pub mod memory;
pub mod memory_pools;
pub mod occupancy;
pub mod peer;
pub mod result;
pub mod slice;
pub mod stream;
pub mod unified;
