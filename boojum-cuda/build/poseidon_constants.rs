use boojum::field::goldilocks::GoldilocksField;
use boojum::field::U64Representable;
use boojum::implementations::poseidon_goldilocks_params::*;

// use itertools::Itertools;

const TEMPLATE_PATH: &str = "native/poseidon_constants_template.cuh";
const RESULT_PATH: &str = "native/poseidon_constants.cuh";

fn split_u64(value: u64) -> (u32, u32) {
    let lo = value as u32;
    let hi = (value >> 32) as u32;
    (lo, hi)
}

// fn get_usize_array_string(values: impl Iterator<Item = usize>) -> String {
//     let mut result = String::new();
//     for x in values {
//         result.push_str(format!("{x},").as_str());
//     }
//     result
// }

fn get_field_array_string(values: &[GoldilocksField]) -> String {
    let mut result = String::new();
    for x in values {
        let (lo, hi) = split_u64(x.as_u64());
        result.push_str(format!("{{{lo:#010x},{hi:#010x}}},").as_str());
    }
    result
}

fn get_field_2d_array_string<const COUNT: usize>(values: &[[GoldilocksField; COUNT]]) -> String {
    let mut result = String::from('\n');
    for row in values {
        result.push_str("  {");
        result.push_str(get_field_array_string(row).as_str());
        result.push_str("},\n");
    }
    result
}

// fn get_mds_matrix_exps() -> String {
//     let values = MDS_MATRIX_EXPS;
//     assert_eq!(values.len(), STATE_WIDTH);
//     get_usize_array_string(values.into_iter())
// }
//
// fn get_mds_matrix_exps_order() -> String {
//     let values = MDS_MATRIX_EXPS;
//     assert_eq!(values.len(), STATE_WIDTH);
//     get_usize_array_string((0..STATE_WIDTH).sorted_by_key(|i| values[*i]).rev())
// }
//
// fn get_mds_matrix_shifts() -> String {
//     let values = MDS_MATRIX_EXPS;
//     assert_eq!(values.len(), STATE_WIDTH);
//     get_usize_array_string(
//         values
//             .iter()
//             .sorted()
//             .rev()
//             .tuple_windows()
//             .map(|(a, b)| a - b)
//             .chain(0usize..1usize),
//     )
// }

fn get_all_round_constants() -> String {
    let values = ALL_ROUND_CONSTANTS_AS_FIELD_ELEMENTS;
    assert_eq!(values.len(), STATE_WIDTH * TOTAL_NUM_ROUNDS);
    let chunks: Vec<[GoldilocksField; STATE_WIDTH]> = values
        .chunks(STATE_WIDTH)
        .map(|c| c.try_into().unwrap())
        .collect();
    get_field_2d_array_string(&chunks)
}

// fn get_fused_round_constants() -> String {
//     let values = ROUND_CONSTANTS_FUZED_LAST_FULL_AND_FIRST_PARTIAL;
//     assert_eq!(values.len(), STATE_WIDTH);
//     get_field_array_string(&values)
// }
//
// fn get_fused_dense_matrix_constants() -> String {
//     let values = FUZED_DENSE_MATRIX_LAST_FULL_AND_FIRST_PARTIAL;
//     assert_eq!(values.len(), STATE_WIDTH);
//     assert_eq!(values[0].len(), STATE_WIDTH);
//     get_field_2d_array_string(&values)
// }
//
// fn get_fused_s_boxes_constants() -> String {
//     let values = ROUND_CONSTANTS_FOR_FUZED_SBOXES;
//     assert_eq!(values.len(), NUM_PARTIAL_ROUNDS);
//     get_field_array_string(&values)
// }
//
// fn get_vs_constants() -> String {
//     let values = VS_FOR_PARTIAL_ROUNDS;
//     assert_eq!(values.len(), NUM_PARTIAL_ROUNDS);
//     assert_eq!(values[0].len(), STATE_WIDTH - 1);
//     get_field_2d_array_string(&values)
// }
//
// fn get_w_hats_constants() -> String {
//     let values = W_HATS_FOR_PARTIAL_ROUNDS;
//     assert_eq!(values.len(), NUM_PARTIAL_ROUNDS);
//     assert_eq!(values[0].len(), STATE_WIDTH - 1);
//     get_field_2d_array_string(&values)
// }

pub(super) fn generate() {
    let replacements = [
        ("RATE", RATE.to_string()),
        ("CAPACITY", CAPACITY.to_string()),
        ("HALF_NUM_FULL_ROUNDS", HALF_NUM_FULL_ROUNDS.to_string()),
        ("NUM_PARTIAL_ROUNDS", NUM_PARTIAL_ROUNDS.to_string()),
        // ("MDS_MATRIX_EXPS", get_mds_matrix_exps()),
        // ("MDS_MATRIX_EXPS_ORDER", get_mds_matrix_exps_order()),
        // ("MDS_MATRIX_SHIFTS", get_mds_matrix_shifts()),
        ("ALL_ROUND_CONSTANTS", get_all_round_constants()),
        // (
        //     "ROUND_CONSTANTS_FUSED_LAST_FULL_AND_FIRST_PARTIAL",
        //     get_fused_round_constants(),
        // ),
        // (
        //     "FUSED_DENSE_MATRIX_LAST_FULL_AND_FIRST_PARTIAL",
        //     get_fused_dense_matrix_constants(),
        // ),
        // (
        //     "ROUND_CONSTANTS_FOR_FUSED_S_BOXES",
        //     get_fused_s_boxes_constants(),
        // ),
        // ("VS_FOR_PARTIAL_ROUNDS", get_vs_constants()),
        // ("W_HATS_FOR_PARTIAL_ROUNDS", get_w_hats_constants()),
    ];
    super::template::generate(&replacements, TEMPLATE_PATH, RESULT_PATH);
}
