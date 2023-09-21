#![feature(custom_test_frameworks)]
#![test_runner(criterion::runner)]

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use boojum_cuda::blake2s::blake2s_pow;
use criterion_cuda::CudaMeasurement;
use cudart::memory::{memory_set_async, DeviceAllocation};
use cudart::stream::CudaStream;

fn blake2s(c: &mut Criterion<CudaMeasurement>) {
    const MIN_BITS_COUNT: u32 = 17;
    const MAX_BITS_COUNT: u32 = 32;
    let mut d_seed = DeviceAllocation::alloc(32).unwrap();
    let mut d_result = DeviceAllocation::alloc(1).unwrap();
    let stream = CudaStream::default();
    memory_set_async(&mut d_seed, 42, &stream).unwrap();
    let mut group = c.benchmark_group("blake2s");
    for bits_count in MIN_BITS_COUNT..=MAX_BITS_COUNT {
        let max_nonce = 1 << bits_count;
        group.throughput(Throughput::Elements(max_nonce));
        group.bench_function(BenchmarkId::from_parameter(bits_count), |b| {
            b.iter(|| {
                blake2s_pow(&d_seed, u32::MAX, max_nonce, &mut d_result[0], &stream).unwrap();
            })
        });
    }
    group.finish();
}

criterion_group!(
    name = bench_blake2s;
    config = Criterion::default().with_measurement::<CudaMeasurement>(CudaMeasurement{});
    targets = blake2s
);
criterion_main!(bench_blake2s);
