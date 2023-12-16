#![feature(custom_test_frameworks)]
#![test_runner(criterion::runner)]

use std::mem;
use std::time::Duration;

use boojum::field::goldilocks::GoldilocksField;
use boojum::implementations::poseidon_goldilocks_params::*;
use criterion::{
    criterion_group, criterion_main, BenchmarkId, Criterion, SamplingMode, Throughput,
};
use rand::{thread_rng, Rng};
use rayon::prelude::*;

use boojum_cuda::poseidon::*;
use criterion_cuda::CudaMeasurement;
use cudart::memory::{memory_copy, DeviceAllocation};
use cudart::result::CudaResult;
use cudart::slice::DeviceSlice;
use cudart::stream::CudaStream;

#[allow(clippy::type_complexity)]
fn leaves_group(
    c: &mut Criterion<CudaMeasurement>,
    group_name: String,
    launch: fn(
        &DeviceSlice<GoldilocksField>,
        &mut DeviceSlice<GoldilocksField>,
        u32,
        bool,
        bool,
        &CudaStream,
    ) -> CudaResult<()>,
) {
    const MIN_LOG_N: usize = 17;
    const MAX_LOG_N: usize = 20;
    const CHUNKS_PER_LEAF: usize = 16;
    let mut initialized = false;
    let mut values_device =
        DeviceAllocation::<GoldilocksField>::alloc((CHUNKS_PER_LEAF * RATE) << MAX_LOG_N).unwrap();
    let mut results_device =
        DeviceAllocation::<GoldilocksField>::alloc(CAPACITY << MAX_LOG_N).unwrap();
    let stream = CudaStream::default();
    let mut group = c.benchmark_group(group_name);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));
    group.sampling_mode(SamplingMode::Flat);
    for log_count in MIN_LOG_N..=MAX_LOG_N {
        let bytes = (CHUNKS_PER_LEAF * RATE * mem::size_of::<GoldilocksField>()) << log_count;
        group.throughput(Throughput::Bytes(bytes as u64));
        group.bench_function(BenchmarkId::from_parameter(log_count), |b| {
            if !initialized {
                let values_host: Vec<GoldilocksField> = (0..values_device.len())
                    .into_par_iter()
                    .map(|_| GoldilocksField::from_nonreduced_u64(thread_rng().gen()))
                    .collect();
                memory_copy(&mut values_device, &values_host).unwrap();
                initialized = true;
            }
            b.iter(|| {
                let values = &values_device[..(CHUNKS_PER_LEAF * RATE) << log_count];
                let results = &mut results_device[..CAPACITY << log_count];
                launch(values, results, 0, false, false, &stream).unwrap();
            })
        });
    }
    group.finish();
    stream.destroy().unwrap();
    results_device.free().unwrap();
    values_device.free().unwrap();
}

fn poseidon_single_thread_leaves(c: &mut Criterion<CudaMeasurement>) {
    leaves_group(
        c,
        String::from("poseidon_single_thread_leaves"),
        launch_single_thread_leaves_kernel::<Poseidon>,
    );
}

fn poseidon2_single_thread_leaves(c: &mut Criterion<CudaMeasurement>) {
    leaves_group(
        c,
        String::from("poseidon2_single_thread_leaves"),
        launch_single_thread_leaves_kernel::<Poseidon2>,
    );
}

fn poseidon2_cooperative_leaves(c: &mut Criterion<CudaMeasurement>) {
    leaves_group(
        c,
        String::from("poseidon2_cooperative_leaves"),
        launch_cooperative_leaves_kernel::<Poseidon2>,
    );
}

fn nodes_group(
    c: &mut Criterion<CudaMeasurement>,
    group_name: String,
    launch: fn(
        &DeviceSlice<GoldilocksField>,
        &mut DeviceSlice<GoldilocksField>,
        &CudaStream,
    ) -> CudaResult<()>,
) {
    const MIN_LOG_N: usize = 4;
    const MAX_LOG_N: usize = 23;
    let mut initialized = false;
    let mut values_device = DeviceAllocation::<GoldilocksField>::alloc(RATE << MAX_LOG_N).unwrap();
    let mut results_device =
        DeviceAllocation::<GoldilocksField>::alloc(CAPACITY << MAX_LOG_N).unwrap();
    let stream = CudaStream::default();
    let mut group = c.benchmark_group(group_name);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_millis(2500));
    group.sampling_mode(SamplingMode::Flat);
    for log_count in MIN_LOG_N..=MAX_LOG_N {
        let bytes = (RATE * mem::size_of::<GoldilocksField>()) << log_count;
        group.throughput(Throughput::Bytes(bytes as u64));
        group.bench_function(BenchmarkId::from_parameter(log_count), |b| {
            if !initialized {
                let values_host: Vec<GoldilocksField> = (0..values_device.len())
                    .into_par_iter()
                    .map(|_| GoldilocksField::from_nonreduced_u64(thread_rng().gen()))
                    .collect();
                memory_copy(&mut values_device, &values_host).unwrap();
                initialized = true;
            }
            b.iter(|| {
                let values = &values_device[..RATE << log_count];
                let results = &mut results_device[..CAPACITY << log_count];
                launch(values, results, &stream).unwrap();
            })
        });
    }
    group.finish();
    stream.destroy().unwrap();
    results_device.free().unwrap();
    values_device.free().unwrap();
}

fn poseidon_single_thread_nodes(c: &mut Criterion<CudaMeasurement>) {
    nodes_group(
        c,
        String::from("poseidon_single_thread_nodes"),
        launch_single_thread_nodes_kernel::<Poseidon>,
    );
}

fn poseidon_cooperative_nodes(c: &mut Criterion<CudaMeasurement>) {
    nodes_group(
        c,
        String::from("poseidon_cooperative_nodes"),
        launch_cooperative_nodes_kernel::<Poseidon>,
    );
}

fn poseidon2_single_thread_nodes(c: &mut Criterion<CudaMeasurement>) {
    nodes_group(
        c,
        String::from("poseidon2_single_thread_nodes"),
        launch_single_thread_nodes_kernel::<Poseidon2>,
    );
}

fn poseidon2_cooperative_nodes(c: &mut Criterion<CudaMeasurement>) {
    nodes_group(
        c,
        String::from("poseidon2_cooperative_nodes"),
        launch_cooperative_nodes_kernel::<Poseidon2>,
    );
}

fn merkle_tree<PoseidonVariant: PoseidonImpl>(
    c: &mut Criterion<CudaMeasurement>,
    group_name: String,
) {
    const MIN_LOG_N: usize = 17;
    const MAX_LOG_N: usize = 20;
    const CHUNKS_PER_LEAF: usize = 16;
    const LAYER_CAP: u32 = 4;
    let mut initialized = false;
    let mut values_device =
        DeviceAllocation::<GoldilocksField>::alloc((CHUNKS_PER_LEAF * RATE) << MAX_LOG_N).unwrap();
    let mut results_device =
        DeviceAllocation::<GoldilocksField>::alloc(CAPACITY << (MAX_LOG_N + 1)).unwrap();
    let stream = CudaStream::default();
    let mut group = c.benchmark_group(group_name);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));
    group.sampling_mode(SamplingMode::Flat);
    for log_count in MIN_LOG_N..=MAX_LOG_N {
        let bytes = (CHUNKS_PER_LEAF * RATE * mem::size_of::<GoldilocksField>()) << log_count;
        group.throughput(Throughput::Bytes(bytes as u64));
        group.bench_function(BenchmarkId::from_parameter(log_count), |b| {
            if !initialized {
                let values_host: Vec<GoldilocksField> = (0..values_device.len())
                    .into_par_iter()
                    .map(|_| GoldilocksField::from_nonreduced_u64(thread_rng().gen()))
                    .collect();
                memory_copy(&mut values_device, &values_host).unwrap();
                initialized = true;
            }
            b.iter(|| {
                let values = &values_device[..(CHUNKS_PER_LEAF * RATE) << log_count];
                let results = &mut results_device[..CAPACITY << (log_count + 1)];
                let layers_count = log_count as u32 + 1 - LAYER_CAP;
                build_merkle_tree::<PoseidonVariant>(values, results, 0, &stream, layers_count)
                    .unwrap();
            })
        });
    }
    group.finish();
    stream.destroy().unwrap();
    results_device.free().unwrap();
    values_device.free().unwrap();
}

fn poseidon_merkle_tree(c: &mut Criterion<CudaMeasurement>) {
    merkle_tree::<Poseidon>(c, String::from("poseidon_merkle_tree"));
}

fn poseidon2_merkle_tree(c: &mut Criterion<CudaMeasurement>) {
    merkle_tree::<Poseidon2>(c, String::from("poseidon2_merkle_tree"));
}

criterion_group!(
    name = bench_poseidon;
    config = Criterion::default().with_measurement::<CudaMeasurement>(CudaMeasurement{});
    targets = poseidon_single_thread_leaves, poseidon2_single_thread_leaves, poseidon2_cooperative_leaves, poseidon_single_thread_nodes, poseidon_cooperative_nodes, poseidon2_single_thread_nodes, poseidon2_cooperative_nodes, poseidon_merkle_tree, poseidon2_merkle_tree,
);
criterion_main!(bench_poseidon);
