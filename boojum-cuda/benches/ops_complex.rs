#![feature(custom_test_frameworks)]
#![test_runner(criterion::runner)]

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use boojum_cuda::device_structures::DeviceMatrixMut;
use boojum_cuda::ops_complex::bit_reverse_in_place;
use criterion_cuda::CudaMeasurement;
use cudart::memory::DeviceAllocation;
use cudart::stream::CudaStream;

fn bit_reverse(c: &mut Criterion<CudaMeasurement>) {
    const LOG_MIN_BATCH_SIZE: usize = 0;
    const LOG_MAX_BATCH_SIZE: usize = 7;
    const BITS: usize = 20;
    let mut d_values = DeviceAllocation::alloc(1 << (LOG_MAX_BATCH_SIZE + BITS)).unwrap();
    let stream = CudaStream::default();
    let mut group = c.benchmark_group("bit_reverse");
    for log_batch_size in LOG_MIN_BATCH_SIZE..=LOG_MAX_BATCH_SIZE {
        let size = 1 << (log_batch_size + BITS);
        group.throughput(Throughput::Bytes(size as u64 * 8));
        group.bench_function(BenchmarkId::from_parameter(log_batch_size), |b| {
            b.iter(|| {
                let mut matrix = DeviceMatrixMut::new(&mut d_values[..size], 1 << BITS);
                bit_reverse_in_place(&mut matrix, &stream).unwrap();
            })
        });
    }
    group.finish();
}

criterion_group!(
    name = bench_bit_reverse;
    config = Criterion::default().with_measurement::<CudaMeasurement>(CudaMeasurement {});
    targets = bit_reverse
);

criterion_main!(bench_bit_reverse);
