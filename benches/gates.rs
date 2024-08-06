#![feature(custom_test_frameworks)]
#![test_runner(criterion::runner)]

use std::time::Duration;

use boojum::field::goldilocks::GoldilocksField;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, SamplingMode};
use rand::prelude::*;
use rayon::prelude::*;

use era_boojum_cuda::device_structures::{DeviceMatrixChunk, DeviceMatrixChunkMut};
use era_boojum_cuda::gates::*;
use era_criterion_cuda::CudaMeasurement;
use era_cudart::memory::{memory_copy, DeviceAllocation};
use era_cudart::slice::DeviceSlice;
use era_cudart::stream::CudaStream;

fn poseidon_group(c: &mut Criterion<CudaMeasurement>, group_name: &str, gate_name: &str) {
    const VARIABLES_COUNT: usize = 140;
    const CONSTANTS_COUNT: usize = 8;
    const SELECTOR_MASK: u32 = 1;
    const SELECTOR_COUNT: u32 = 1;
    const CHALLENGES_COUNT: usize = 4;
    const MIN_LOG_TRACE_LENGTH: usize = 16;
    const MAX_LOG_TRACE_LENGTH: usize = 23;
    const REPETITIONS_COUNT: u32 = 1;
    const TRACE_WIDTH: usize = VARIABLES_COUNT * REPETITIONS_COUNT as usize + CONSTANTS_COUNT;
    let gate_id = find_gate_id_by_name(gate_name).unwrap();
    let trace_host: Vec<GoldilocksField> = (0..TRACE_WIDTH << MAX_LOG_TRACE_LENGTH)
        .into_par_iter()
        .map(|_| GoldilocksField::from_nonreduced_u64(thread_rng().gen()))
        .collect();
    let mut trace_device =
        DeviceAllocation::<GoldilocksField>::alloc(TRACE_WIDTH << MAX_LOG_TRACE_LENGTH).unwrap();
    memory_copy(&mut trace_device, &trace_host).unwrap();
    let constants_offset = (VARIABLES_COUNT * REPETITIONS_COUNT as usize) << MAX_LOG_TRACE_LENGTH;
    let variable_polys = &trace_device[..constants_offset];
    let witness_polys = DeviceSlice::empty();
    let constant_polys = &trace_device[constants_offset..];
    let challenges_device = DeviceAllocation::alloc(CHALLENGES_COUNT).unwrap();
    let mut quotient_polys_device =
        DeviceAllocation::alloc(CHALLENGES_COUNT << MAX_LOG_TRACE_LENGTH).unwrap();
    let stream = CudaStream::default();
    let params = GateEvaluationParams {
        id: gate_id,
        selector_mask: SELECTOR_MASK,
        selector_count: SELECTOR_COUNT,
        repetitions_count: REPETITIONS_COUNT,
        initial_variables_offset: 0,
        initial_witnesses_offset: 0,
        initial_constants_offset: 0,
        repetition_variables_offset: 130,
        repetition_witnesses_offset: 0,
        repetition_constants_offset: 0,
    };
    let mut group = c.benchmark_group(group_name);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));
    group.sampling_mode(SamplingMode::Flat);
    for log_length in MIN_LOG_TRACE_LENGTH..=MAX_LOG_TRACE_LENGTH {
        group.bench_function(BenchmarkId::from_parameter(log_length), |b| {
            b.iter(|| {
                let rows = 1 << log_length;
                let stride = 1 << MAX_LOG_TRACE_LENGTH;
                let variable_polys = DeviceMatrixChunk::new(variable_polys, stride, 0, rows);
                let witness_polys = DeviceMatrixChunk::new(witness_polys, stride, 0, rows);
                let constant_polys = DeviceMatrixChunk::new(constant_polys, stride, 0, rows);
                let mut quotient_polys =
                    DeviceMatrixChunkMut::new(&mut quotient_polys_device, stride, 0, rows);
                evaluate_gate(
                    &params,
                    &variable_polys,
                    &witness_polys,
                    &constant_polys,
                    &challenges_device,
                    &mut quotient_polys,
                    0,
                    &stream,
                )
                .unwrap();
            })
        });
    }
    group.finish();
    stream.destroy().unwrap();
    trace_device.free().unwrap();
    challenges_device.free().unwrap();
    quotient_polys_device.free().unwrap();
}

fn poseidon2_gate(c: &mut Criterion<CudaMeasurement>) {
    poseidon_group(c, "poseidon2_gate", "poseidon_2_round_function_flattened_evaluator_goldilocks_field_8_12_4_poseidon_2_goldilocks_1451e131831047e6");
}

fn fma_gate(c: &mut Criterion<CudaMeasurement>) {
    const VARIABLES_COUNT: usize = 4;
    const CONSTANTS_COUNT: usize = 8;
    const SELECTOR_MASK: u32 = 1;
    const SELECTOR_COUNT: u32 = 1;
    const CHALLENGES_COUNT: usize = 4;
    const MIN_LOG_TRACE_LENGTH: usize = 16;
    const MAX_LOG_TRACE_LENGTH: usize = 23;
    const REPETITIONS_COUNT: u32 = 35;
    const TRACE_WIDTH: usize = VARIABLES_COUNT * REPETITIONS_COUNT as usize + CONSTANTS_COUNT;
    let gate_id =
        find_gate_id_by_name("fma_gate_in_base_without_constant_constraint_evaluator").unwrap();
    let trace_host: Vec<GoldilocksField> = (0..TRACE_WIDTH << MAX_LOG_TRACE_LENGTH)
        .into_par_iter()
        .map(|_| GoldilocksField::from_nonreduced_u64(thread_rng().gen()))
        .collect();
    let mut trace_device =
        DeviceAllocation::<GoldilocksField>::alloc(TRACE_WIDTH << MAX_LOG_TRACE_LENGTH).unwrap();
    memory_copy(&mut trace_device, &trace_host).unwrap();
    let constants_offset = (VARIABLES_COUNT * REPETITIONS_COUNT as usize) << MAX_LOG_TRACE_LENGTH;
    let variable_polys = &trace_device[..constants_offset];
    let witness_polys = DeviceSlice::empty();
    let constant_polys = &trace_device[constants_offset..];
    let challenges_device = DeviceAllocation::alloc(CHALLENGES_COUNT).unwrap();
    let mut quotient_polys_device =
        DeviceAllocation::alloc(CHALLENGES_COUNT << MAX_LOG_TRACE_LENGTH).unwrap();
    let stream = CudaStream::default();
    let params = GateEvaluationParams {
        id: gate_id,
        selector_mask: SELECTOR_MASK,
        selector_count: SELECTOR_COUNT,
        repetitions_count: REPETITIONS_COUNT,
        initial_variables_offset: 0,
        initial_witnesses_offset: 0,
        initial_constants_offset: 0,
        repetition_variables_offset: 4,
        repetition_witnesses_offset: 0,
        repetition_constants_offset: 0,
    };

    let mut group = c.benchmark_group("fma_gate");
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));
    group.sampling_mode(SamplingMode::Flat);
    for log_length in MIN_LOG_TRACE_LENGTH..=MAX_LOG_TRACE_LENGTH {
        group.bench_function(BenchmarkId::from_parameter(log_length), |b| {
            b.iter(|| {
                let rows = 1 << log_length;
                let stride = 1 << MAX_LOG_TRACE_LENGTH;
                let variable_polys = DeviceMatrixChunk::new(variable_polys, stride, 0, rows);
                let witness_polys = DeviceMatrixChunk::new(witness_polys, stride, 0, rows);
                let constant_polys = DeviceMatrixChunk::new(constant_polys, stride, 0, rows);
                let mut quotient_polys =
                    DeviceMatrixChunkMut::new(&mut quotient_polys_device, stride, 0, rows);
                evaluate_gate(
                    &params,
                    &variable_polys,
                    &witness_polys,
                    &constant_polys,
                    &challenges_device,
                    &mut quotient_polys,
                    0,
                    &stream,
                )
                .unwrap();
            })
        });
    }
    group.finish();
    stream.destroy().unwrap();
    trace_device.free().unwrap();
    challenges_device.free().unwrap();
    quotient_polys_device.free().unwrap();
}

criterion_group!(
    name = bench_gates;
    config = Criterion::default().with_measurement::<CudaMeasurement>(CudaMeasurement{});
    targets = fma_gate, poseidon2_gate
);
criterion_main!(bench_gates);
