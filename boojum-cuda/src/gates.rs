use std::ffi::c_void;
use std::mem::size_of;

use boojum::cs::traits::evaluator::GateConstraintEvaluator;
use boojum::cs::traits::gate::Gate;
use boojum::field::goldilocks::GoldilocksField;
use boojum::gpu_synthesizer::get_evaluator_name;

use cudart::kernel_args;
use cudart::result::{CudaResult, CudaResultWrap};
use cudart::stream::CudaStream;
use cudart_sys::cudaLaunchKernel;

use crate::device_structures::{
    BaseFieldDeviceType, DeviceMatrixChunkImpl, DeviceMatrixChunkMutImpl, DeviceVectorImpl,
    MutPtrAndStride, PtrAndStride, VectorizedExtensionFieldDeviceType,
};
use crate::extension_field::VectorizedExtensionField;
use crate::utils::{get_grid_block_dims_for_threads_count, WARP_SIZE};

#[repr(C)]
#[derive(Clone, Copy)]
pub struct GateEvaluationParams {
    pub id: u32,
    pub selector_mask: u32,
    pub selector_count: u32,
    pub repetitions_count: u32,
    pub initial_variables_offset: u32,
    pub initial_witnesses_offset: u32,
    pub initial_constants_offset: u32,
    pub repetition_variables_offset: u32,
    pub repetition_witnesses_offset: u32,
    pub repetition_constants_offset: u32,
}

macro_rules! kernel_binding {
    ($function_name:ident) => {
        fn $function_name(
            params: GateEvaluationParams,
            variable_polys: *const GoldilocksField,
            witness_polys: *const GoldilocksField,
            constant_polys: *const GoldilocksField,
            challenges: PtrAndStride<VectorizedExtensionFieldDeviceType>,
            quotient_polys: MutPtrAndStride<VectorizedExtensionFieldDeviceType>,
            challenges_count: u32,
            challenges_power_offset: u32,
            rows_count: u32,
            inputs_stride: u32,
        );
    };
}

type Kernel = unsafe extern "C" fn(
    params: GateEvaluationParams,
    variable_polys: *const BaseFieldDeviceType,
    witness_polys: *const BaseFieldDeviceType,
    constant_polys: *const BaseFieldDeviceType,
    challenges: PtrAndStride<VectorizedExtensionFieldDeviceType>,
    quotient_polys: MutPtrAndStride<VectorizedExtensionFieldDeviceType>,
    challenges_count: u32,
    challenges_power_offset: u32,
    rows_count: u32,
    inputs_stride: u32,
);

#[allow(dead_code)]
struct GateData {
    name: &'static str,
    contributions_count: u32,
    max_variable_index: Option<u32>,
    max_witness_index: Option<u32>,
    max_constant_index: Option<u32>,
    kernel: Kernel,
}

include!("gates_data.rs");

pub fn find_gate_id_by_name(name: &str) -> Option<u32> {
    HASH_MAP.get(name).copied()
}

pub fn find_gate_id_for_evaluator<E: GateConstraintEvaluator<GoldilocksField>>(
    evaluator: &E,
) -> Option<u32> {
    let name = get_evaluator_name(evaluator);
    find_gate_id_by_name(&name)
}

pub fn find_gate_id_for_evaluator_type<E: GateConstraintEvaluator<GoldilocksField>>(
    params: E::UniqueParameterizationParams,
) -> Option<u32> {
    let evaluator = E::new_from_parameters(params);
    let name = get_evaluator_name(&evaluator);
    find_gate_id_by_name(&name)
}

pub fn find_gate_id_for_gate_type<G: Gate<GoldilocksField>>(
    params: <<G as Gate<GoldilocksField>>::Evaluator as GateConstraintEvaluator<
        GoldilocksField,
    >>::UniqueParameterizationParams,
) -> Option<u32> {
    find_gate_id_for_evaluator_type::<G::Evaluator>(params)
}

fn get_required_values_count(
    repetitions_count: u32,
    initial_offset: u32,
    repetition_offset: u32,
    max_index: Option<u32>,
) -> u32 {
    max_index
        .map_or(0, |x| x + 1)
        .max(initial_offset + (repetition_offset * repetitions_count))
}

#[allow(clippy::too_many_arguments)]
pub fn evaluate_gate<I, C, Q>(
    params: &GateEvaluationParams,
    variable_polys: &I,
    witness_polys: &I,
    constant_polys: &I,
    challenges: &C,
    quotient_polys: &mut Q,
    challenges_power_offset: u32,
    stream: &CudaStream,
) -> CudaResult<u32>
where
    I: DeviceMatrixChunkImpl<GoldilocksField> + ?Sized,
    C: DeviceVectorImpl<VectorizedExtensionField> + ?Sized,
    Q: DeviceMatrixChunkMutImpl<VectorizedExtensionField> + ?Sized,
{
    let inputs_stride = variable_polys.stride();
    assert_eq!(witness_polys.stride(), inputs_stride);
    assert_eq!(constant_polys.stride(), inputs_stride);
    let rows_count = variable_polys.rows();
    assert_eq!(witness_polys.rows(), rows_count);
    assert_eq!(constant_polys.rows(), rows_count);
    assert_eq!(quotient_polys.rows(), rows_count);
    assert!(params.selector_count <= 32);
    let GateData {
        name: _,
        contributions_count: writes_count,
        max_variable_index,
        max_witness_index,
        max_constant_index,
        kernel,
    } = get_gate_data(params.id);
    assert!(variable_polys.cols() <= u32::MAX as usize);
    let variable_polys_count = variable_polys.cols() as u32;
    let required_values_count = get_required_values_count(
        params.repetitions_count,
        params.initial_variables_offset,
        params.repetition_variables_offset,
        max_variable_index,
    );
    assert!(variable_polys_count >= required_values_count);
    assert!(witness_polys.cols() <= u32::MAX as usize);
    let witness_polys_count = witness_polys.cols() as u32;
    let required_witnesses_count = get_required_values_count(
        params.repetitions_count,
        params.initial_witnesses_offset,
        params.repetition_witnesses_offset,
        max_witness_index,
    );
    assert!(witness_polys_count >= required_witnesses_count);
    assert!(constant_polys.cols() <= u32::MAX as usize);
    let constant_polys_count = constant_polys.cols() as u32;
    let required_constants_count = get_required_values_count(
        params.repetitions_count,
        params.initial_constants_offset + params.selector_count,
        params.repetition_constants_offset,
        max_constant_index,
    );
    assert!(constant_polys_count >= required_constants_count);
    let challenges_count = challenges.slice().len();
    assert_eq!(challenges_count, quotient_polys.cols());
    assert!(challenges_count <= u32::MAX as usize);
    let challenges_count = challenges_count as u32;
    assert!(rows_count <= u32::MAX as usize);
    let rows_count = rows_count as u32;
    assert!(challenges_count <= rows_count);
    unsafe {
        let variable_polys = variable_polys.as_ptr();
        let witness_polys = witness_polys.as_ptr();
        let constant_polys = constant_polys.as_ptr();
        let challenges = challenges.as_ptr_and_stride();
        let quotient_polys = quotient_polys.as_mut_ptr_and_stride();
        let (grid_dim, block_dim) =
            get_grid_block_dims_for_threads_count(WARP_SIZE * 4, rows_count);
        let mut args = kernel_args!(
            params,
            &variable_polys,
            &witness_polys,
            &constant_polys,
            &challenges,
            &quotient_polys,
            &challenges_count,
            &challenges_power_offset,
            &rows_count,
            &inputs_stride,
        );
        let shared_size =
            size_of::<VectorizedExtensionField>() * (challenges_count * (2 + block_dim.x)) as usize;
        cudaLaunchKernel(
            kernel as *const c_void,
            grid_dim,
            block_dim,
            args.as_mut_ptr(),
            shared_size,
            stream.into(),
        )
        .wrap_value(challenges_power_offset + writes_count * params.repetitions_count)
    }
}

#[allow(clippy::too_many_arguments)]
pub fn evaluate_gates<I, C, Q>(
    gates_params: &[GateEvaluationParams],
    variable_polys: &I,
    witness_polys: &I,
    constant_polys: &I,
    challenges: &C,
    quotient_polys: &mut Q,
    challenges_power_offset: u32,
    stream: &CudaStream,
) -> CudaResult<u32>
where
    I: DeviceMatrixChunkImpl<GoldilocksField> + ?Sized,
    C: DeviceVectorImpl<VectorizedExtensionField> + ?Sized,
    Q: DeviceMatrixChunkMutImpl<VectorizedExtensionField> + ?Sized,
{
    let mut challenges_power_offset = challenges_power_offset;
    for params in gates_params {
        challenges_power_offset = evaluate_gate(
            params,
            variable_polys,
            witness_polys,
            constant_polys,
            challenges,
            quotient_polys,
            challenges_power_offset,
            stream,
        )?;
    }
    Ok(challenges_power_offset)
}

#[cfg(test)]
mod tests {
    use boojum::algebraic_props::poseidon2_parameters::{
        Poseidon2GoldilocksExternalMatrix, Poseidon2GoldilocksInnerMatrix,
    };
    use boojum::cs::gates::*;
    use boojum::cs::traits::evaluator::{
        GateConstraintEvaluator, GenericDynamicEvaluatorOverGeneralPurposeColumns,
        GenericRowwiseEvaluator,
    };
    use boojum::cs::traits::gate::Gate;
    use boojum::cs::CSGeometry;
    use boojum::field::goldilocks::{GoldilocksExt2, GoldilocksField};
    use boojum::field::Field;
    use boojum::gpu_synthesizer::{TestDestination, TestSource};
    use boojum::implementations::poseidon2::Poseidon2Goldilocks;
    use rand::prelude::*;

    use cudart::memory::{memory_copy_async, DeviceAllocation};
    use cudart::slice::DeviceSlice;

    use crate::device_structures::{DeviceMatrix, DeviceMatrixMut};
    use crate::extension_field::ExtensionField;
    use crate::ops_simple::set_to_zero;

    use super::*;

    type F = GoldilocksField;
    type EF = ExtensionField;

    fn copy_vectors_async(
        dst: &mut DeviceSlice<F>,
        src: &Vec<Vec<F>>,
        stream: &CudaStream,
        len: usize,
    ) {
        let mut offset = 0usize;
        assert_eq!(dst.len(), len * src.len());
        for src in src {
            assert_eq!(src.len(), len);
            memory_copy_async(&mut dst[offset..offset + len], src, stream).unwrap();
            offset += len;
        }
    }

    fn test_gate<G: Gate<F>>(
        params: <<G as Gate<F>>::Evaluator as GateConstraintEvaluator<
            F,
        >>::UniqueParameterizationParams,
    ) {
        const VARIABLES_COUNT: usize = 144;
        const WITNESSES_COUNT: usize = 16;
        const SELECTORS_COUNT: usize = 4;
        const CONSTANTS_COUNT: usize = 16;
        const LOG_TRACE_LENGTH: usize = 10;
        const CHALLENGES_COUNT: usize = 4;
        const CHALLENGES_POWER_OFFSET: u32 = 42;
        const TRACE_LENGTH: usize = 1 << LOG_TRACE_LENGTH;
        let geometry = CSGeometry {
            num_columns_under_copy_permutation: VARIABLES_COUNT,
            num_witness_columns: WITNESSES_COUNT,
            num_constant_columns: CONSTANTS_COUNT - SELECTORS_COUNT,
            max_allowed_constraint_degree: <usize>::MAX,
        };
        let mut source = TestSource::<F>::random_source(
            VARIABLES_COUNT,
            WITNESSES_COUNT,
            CONSTANTS_COUNT,
            TRACE_LENGTH,
        );
        source
            .constants
            .iter_mut()
            .take(SELECTORS_COUNT)
            .for_each(|v| v.fill(F::ZERO));
        let evaluator = G::Evaluator::new_from_parameters(params);
        let ctx = &mut ();
        let gate_id = find_gate_id_for_evaluator(&evaluator).unwrap();
        let num_repetitions = G::Evaluator::num_repetitions_in_geometry(&evaluator, &geometry);
        let per_chunk_offset =
            G::Evaluator::per_chunk_offset_for_repetition_over_general_purpose_columns(&evaluator);
        let global_constants = evaluator.create_global_constants::<F>(ctx);
        let evaluator = GenericRowwiseEvaluator::<F, F, _> {
            evaluator,
            global_constants,
            num_repetitions,
            per_chunk_offset,
        };
        let num_terms = G::Evaluator::num_quotient_terms();
        let mut destination = TestDestination::<F>::new(TRACE_LENGTH, num_terms * num_repetitions);
        for _ in 0..TRACE_LENGTH {
            evaluator.evaluate_over_general_purpose_columns(
                &mut source,
                &mut destination,
                SELECTORS_COUNT,
                ctx,
            );
        }
        let stream = CudaStream::default();
        let mut trace_device = DeviceAllocation::<F>::alloc(
            (VARIABLES_COUNT + WITNESSES_COUNT + CONSTANTS_COUNT) << LOG_TRACE_LENGTH,
        )
        .unwrap();
        let mut challenges_device = DeviceAllocation::<F>::alloc(CHALLENGES_COUNT << 1).unwrap();
        let mut quotient_polys_device =
            DeviceAllocation::<F>::alloc(CHALLENGES_COUNT << (LOG_TRACE_LENGTH + 1)).unwrap();
        const WITNESSES_OFFSET: usize = VARIABLES_COUNT << LOG_TRACE_LENGTH;
        const CONSTANTS_OFFSET: usize = WITNESSES_OFFSET + (WITNESSES_COUNT << LOG_TRACE_LENGTH);
        copy_vectors_async(
            &mut trace_device[..WITNESSES_OFFSET],
            &source.variables,
            &stream,
            TRACE_LENGTH,
        );
        copy_vectors_async(
            &mut trace_device[WITNESSES_OFFSET..CONSTANTS_OFFSET],
            &source.witness,
            &stream,
            TRACE_LENGTH,
        );
        copy_vectors_async(
            &mut trace_device[CONSTANTS_OFFSET..],
            &source.constants,
            &stream,
            TRACE_LENGTH,
        );
        let challenges_host = (0..CHALLENGES_COUNT * 2)
            .map(|_| F::from_nonreduced_u64(thread_rng().gen()))
            .collect::<Vec<_>>();
        memory_copy_async(&mut challenges_device, &challenges_host, &stream).unwrap();
        set_to_zero(&mut quotient_polys_device, &stream).unwrap();
        let mut quotient_polys_host = vec![F::ZERO; CHALLENGES_COUNT << (LOG_TRACE_LENGTH + 1)];
        let params = GateEvaluationParams {
            id: gate_id,
            selector_mask: 0,
            selector_count: SELECTORS_COUNT as u32,
            repetitions_count: num_repetitions as u32,
            initial_variables_offset: 0,
            initial_witnesses_offset: 0,
            initial_constants_offset: 0,
            repetition_variables_offset: per_chunk_offset.variables_offset as u32,
            repetition_witnesses_offset: per_chunk_offset.witnesses_offset as u32,
            repetition_constants_offset: per_chunk_offset.constants_offset as u32,
        };
        let variable_polys = DeviceMatrix::new(&trace_device[..WITNESSES_OFFSET], TRACE_LENGTH);
        let witness_polys = DeviceMatrix::new(
            &trace_device[WITNESSES_OFFSET..CONSTANTS_OFFSET],
            TRACE_LENGTH,
        );
        let constant_polys = DeviceMatrix::new(&trace_device[CONSTANTS_OFFSET..], TRACE_LENGTH);
        let challenges = unsafe { challenges_device.transmute::<VectorizedExtensionField>() };
        let mut quotient_polys = DeviceMatrixMut::new(
            unsafe { quotient_polys_device.transmute_mut::<VectorizedExtensionField>() },
            TRACE_LENGTH,
        );
        let challenges_power_offset = evaluate_gate(
            &params,
            &variable_polys,
            &witness_polys,
            &constant_polys,
            challenges,
            &mut quotient_polys,
            CHALLENGES_POWER_OFFSET,
            &stream,
        )
        .unwrap();
        assert_eq!(
            challenges_power_offset,
            CHALLENGES_POWER_OFFSET + (num_terms * num_repetitions) as u32
        );
        memory_copy_async(&mut quotient_polys_host, &quotient_polys_device, &stream).unwrap();
        stream.synchronize().unwrap();
        for row in 0..TRACE_LENGTH {
            let mut challenges = [EF::ZERO; CHALLENGES_COUNT];
            let mut sums = [EF::ZERO; CHALLENGES_COUNT];
            let mut powers = [EF::ZERO; CHALLENGES_COUNT];
            for i in 0..CHALLENGES_COUNT {
                let c0 = challenges_host[i];
                let c1 = challenges_host[i + CHALLENGES_COUNT];
                let v = EF::from_coeff_in_base([c0, c1]);
                challenges[i] = v;
                powers[i] = v.pow_u64(CHALLENGES_POWER_OFFSET as u64);
            }
            for repetition in 0..num_repetitions {
                for term in 0..num_terms {
                    for challenge in 0..CHALLENGES_COUNT {
                        let mut p = powers[challenge];
                        p.mul_assign_by_base(
                            &destination.terms[repetition * num_terms + term][row],
                        );
                        sums[challenge].add_assign(&p);
                        powers[challenge].mul_assign(&challenges[challenge]);
                    }
                }
            }
            for challenge in 0..CHALLENGES_COUNT {
                let c0 = quotient_polys_host[challenge * 2 * TRACE_LENGTH + row];
                let c1 = quotient_polys_host[(challenge * 2 + 1) * TRACE_LENGTH + row];
                let v = EF::from_coeff_in_base([c0, c1]);
                assert_eq!(v, sums[challenge]);
            }
        }
    }

    #[test]
    fn boolean_constraint() {
        test_gate::<BooleanConstraintGate>(());
    }

    #[test]
    fn conditional_swap_1() {
        test_gate::<ConditionalSwapGate<1>>(());
    }

    #[test]
    fn conditional_swap_4() {
        test_gate::<ConditionalSwapGate<4>>(());
    }

    #[test]
    fn constants_allocator() {
        test_gate::<ConstantsAllocatorGate<F>>(());
    }

    #[test]
    fn dot_product_4() {
        test_gate::<DotProductGate<4>>(());
    }

    #[test]
    fn fma_in_base_field_without_constant() {
        test_gate::<FmaGateInBaseFieldWithoutConstant<F>>(());
    }

    #[test]
    fn fma_in_extension_field_without_constant() {
        test_gate::<FmaGateInExtensionWithoutConstant<F, GoldilocksExt2>>(());
    }

    #[test]
    fn matrix_multiplication_poseidon_2_external() {
        test_gate::<MatrixMultiplicationGate<F, 12, Poseidon2GoldilocksExternalMatrix>>(());
    }

    #[test]
    fn matrix_multiplication_poseidon_2_inner() {
        test_gate::<MatrixMultiplicationGate<F, 12, Poseidon2GoldilocksInnerMatrix>>(());
    }

    #[test]
    fn parallel_selection_4() {
        test_gate::<ParallelSelectionGate<4>>(());
    }

    #[test]
    fn poseidon_2_130_0() {
        test_gate::<Poseidon2FlattenedGate<F, 8, 12, 4, Poseidon2Goldilocks>>((130, 0));
    }

    #[test]
    fn quadratic_combination_4() {
        test_gate::<QuadraticCombinationGate<4>>(());
    }

    #[test]
    fn reduction_by_powers_4() {
        test_gate::<ReductionByPowersGate<F, 4>>(());
    }

    #[test]
    fn reduction_4() {
        test_gate::<ReductionGate<F, 4>>(());
    }

    #[test]
    fn selection() {
        test_gate::<SelectionGate>(());
    }

    #[test]
    fn simple_nonlinearity_7() {
        test_gate::<SimpleNonlinearityGate<F, 7>>(());
    }

    #[test]
    fn u32_add() {
        test_gate::<U32AddGate>(());
    }

    #[test]
    fn u8x4_fma() {
        test_gate::<U8x4FMAGate>(());
    }

    #[test]
    fn u32_sub() {
        test_gate::<U32SubGate>(());
    }

    #[test]
    fn u32_tri_add_carry_as_chunk() {
        test_gate::<U32TriAddCarryAsChunkGate>(());
    }

    #[test]
    fn uint_x_add_32() {
        test_gate::<UIntXAddGate<32>>(());
    }

    #[test]
    fn zero_check_false() {
        test_gate::<ZeroCheckGate>(false);
    }

    #[test]
    fn zero_check_true() {
        test_gate::<ZeroCheckGate>(true);
    }
}
