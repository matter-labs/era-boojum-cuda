#![feature(custom_test_frameworks)]
#![test_runner(criterion::runner)]

use std::ops::Range;
use std::time::Duration;

use boojum::field::goldilocks::GoldilocksField;
use criterion::{criterion_group, criterion_main, Criterion, SamplingMode, Throughput};
use rand::{thread_rng, Rng};
use rayon::prelude::*;

use era_boojum_cuda::context::Context;
use era_boojum_cuda::ntt::*;
use era_criterion_cuda::CudaMeasurement;
use era_cudart::memory::{memory_copy, DeviceAllocation};
use era_cudart::stream::CudaStream;

type CudaMeasurementInvElems = CudaMeasurement<true>;

// it's debatable how we should split configs into different ids
// in the same #[criterion] or different #[criterion]s
fn case(
    c: &mut Criterion<CudaMeasurementInvElems>,
    group_name: &str,
    log_n_range: Range<u32>,
    log_extension_degree: u32,
    coset_index: u32,
) {
    const WARM_UP_TIME_MS: u64 = 500;
    const MEASUREMENT_TIME_MS: u64 = 2500;
    const SAMPLE_SIZE: usize = 10;
    // REPS is tailored to increase the time per bench_function sample,
    // while roughly ensuring SAMPLE_SIZE samples can happen within MEASUREMENT_TIME_MS.
    const REPS: u64 = 10;
    let ctx = Context::create(12, 12).unwrap();
    let mut initialized = false;
    // A problem size (num ntts * size per ntt) of 2 GiB should be enough to saturate.
    // We'll hold the problem size fixed across log_n_range.
    let problem_size: usize = 1 << 28; // 8 * 2^28 = 1 GiB
    let mut max_inputs_matrix_device =
        DeviceAllocation::<GoldilocksField>::alloc(problem_size).unwrap();
    let mut max_outputs_matrix_device =
        DeviceAllocation::<GoldilocksField>::alloc(problem_size).unwrap();
    let stream = CudaStream::default();
    let mut group = c.benchmark_group(group_name);
    group.sample_size(SAMPLE_SIZE);
    group.warm_up_time(Duration::from_millis(WARM_UP_TIME_MS));
    group.measurement_time(Duration::from_millis(MEASUREMENT_TIME_MS));
    group.sampling_mode(SamplingMode::Flat);
    for inverse in [false, true] {
        for bitrev_inputs in [false, true] {
            for log_count in log_n_range.clone() {
                let count: u32 = 1 << log_count;
                let num_ntts: u32 = problem_size as u32 >> log_count;
                let stride_between_input_arrays = count;
                let stride_between_output_arrays = count;
                let inputs_matrix_size = (stride_between_input_arrays * num_ntts) as usize;
                let outputs_matrix_size = (stride_between_output_arrays * num_ntts) as usize;
                // Report the time per individual NTT
                group.throughput(Throughput::Elements(REPS * num_ntts as u64));
                let mut id = String::from(if inverse { "inverse, " } else { "forward, " });
                id += if bitrev_inputs { "b2n, " } else { "n2b, " };
                id += "size 2^";
                id += &log_count.to_string();
                group.bench_function(id, |b| {
                    if !initialized {
                        let max_inputs_matrix_host: Vec<GoldilocksField> = (0
                            ..max_inputs_matrix_device.len())
                            .into_par_iter()
                            .map(|_| GoldilocksField::from_nonreduced_u64(thread_rng().gen()))
                            .collect();
                        memory_copy(&mut max_inputs_matrix_device, &max_inputs_matrix_host)
                            .unwrap();
                        initialized = true;
                    }
                    b.iter(|| {
                        let inputs_matrix_device = &max_inputs_matrix_device[..inputs_matrix_size];
                        let outputs_matrix_device =
                            &mut max_outputs_matrix_device[..outputs_matrix_size];
                        for _ in 0..REPS {
                            batch_ntt_out_of_place(
                                inputs_matrix_device,
                                outputs_matrix_device,
                                log_count,
                                num_ntts,
                                0,
                                0,
                                stride_between_input_arrays,
                                stride_between_output_arrays,
                                bitrev_inputs,
                                inverse,
                                log_extension_degree,
                                coset_index,
                                &stream,
                            )
                            .unwrap();
                        }
                    })
                });
            }
        }
    }
    group.finish();
    stream.destroy().unwrap();
    max_inputs_matrix_device.free().unwrap();
    max_outputs_matrix_device.free().unwrap();
    ctx.destroy().unwrap();
}

fn batch_ntt(c: &mut Criterion<CudaMeasurementInvElems>) {
    case(c, "batch_ntt", 16..25, 0, 0);
}

fn batch_lde(c: &mut Criterion<CudaMeasurementInvElems>) {
    case(c, "batch_lde", 16..23, 2, 1);
}

criterion_group!(
    name = bench_ntt;
    config = Criterion::default().with_measurement::<CudaMeasurementInvElems>(CudaMeasurementInvElems{});
    targets = batch_ntt, batch_lde
);
criterion_main!(bench_ntt);
