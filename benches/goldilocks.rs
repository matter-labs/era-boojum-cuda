#![feature(custom_test_frameworks)]
#![test_runner(criterion::runner)]

use std::mem;
use std::time::Duration;

use boojum::field::goldilocks::GoldilocksField;
use criterion::{
    criterion_group, criterion_main, BenchmarkId, Criterion, SamplingMode, Throughput,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use boojum_cuda::ops_simple;
use criterion_cuda::CudaMeasurement;
use cudart::memory::{memory_copy, DeviceAllocation};
use cudart::stream::CudaStream;

fn goldilocks_inv(c: &mut Criterion<CudaMeasurement>) {
    const MIN_LOG_N: usize = 17;
    const MAX_LOG_N: usize = 24;
    const MAX_N: usize = 1usize << MAX_LOG_N;
    // The inv algorithm's per-thread work is highly value-dependent.
    // In other words, the kernel's runtime varies based on input values,
    // and the variance is more significant for smaller sizes.
    // In theory, we could reduce the variance with a more complicated
    // (e.g. cooperative) kernel, if it becomes a bottleneck.
    // For now, standardizing RNG at least ensures our benches give
    // consistent timings across invocations of cargo bench, even though
    // there's still some variance between different sizes.
    let mut rng = StdRng::seed_from_u64(42);
    let values_host: Vec<GoldilocksField> = (0..MAX_N)
        .map(|_| GoldilocksField::from_nonreduced_u64(rng.gen()))
        .collect();
    let mut values_device = DeviceAllocation::<GoldilocksField>::alloc(MAX_N).unwrap();
    memory_copy(&mut values_device, &values_host).unwrap();
    let mut results_device = DeviceAllocation::<GoldilocksField>::alloc(MAX_N).unwrap();
    let stream = CudaStream::default();
    let mut group = c.benchmark_group("goldilocks_inv");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_millis(2500));
    group.sampling_mode(SamplingMode::Flat);
    for log_count in MIN_LOG_N..=MAX_LOG_N {
        let count = 1 << log_count;
        let bytes = mem::size_of::<GoldilocksField>() << log_count;
        group.throughput(Throughput::Bytes(bytes as u64));
        group.bench_function(BenchmarkId::from_parameter(log_count), |b| {
            b.iter(|| {
                let values = &values_device[..count];
                let results = &mut results_device[..count];
                ops_simple::inv(values, results, &stream).unwrap();
            })
        });
    }
    group.finish();
    stream.destroy().unwrap();
    results_device.free().unwrap();
    values_device.free().unwrap();
}

criterion_group!(
    name = bench_goldilocks;
    config = Criterion::default().with_measurement::<CudaMeasurement>(CudaMeasurement{});
    targets = goldilocks_inv
);
criterion_main!(bench_goldilocks);
