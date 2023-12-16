use std::collections::HashSet;
use std::fmt::Display;

use boojum::algebraic_props::poseidon2_parameters::{
    Poseidon2GoldilocksExternalMatrix, Poseidon2GoldilocksInnerMatrix,
};
use boojum::cs::gates::poseidon2::*;
use boojum::cs::gates::*;
use boojum::cs::traits::evaluator::GateConstraintEvaluator;
use boojum::cs::traits::gate::Gate;
use boojum::field::goldilocks::{GoldilocksExt2, GoldilocksField};
use boojum::field::U64Representable;
use boojum::gpu_synthesizer::*;
use boojum::implementations::poseidon2::Poseidon2Goldilocks;
use itertools::Itertools;

use crate::gates::GateType::Generic;

type F = GoldilocksField;

#[derive(Copy, Clone, Debug, PartialEq)]
enum GateType {
    Generic,
    Poseidon2(usize, usize),
    Poseidon2ExternalMatrix,
    Poseidon2InternalMatrix,
}

struct Description {
    capture: GPUDataCapture,
    gate_type: GateType,
}

impl Description {
    fn new<G: Gate<F>>(
        params: <<G as Gate<F>>::Evaluator as GateConstraintEvaluator<F>>::UniqueParameterizationParams,
        gate_type: GateType,
    ) -> Description {
        let ctx = &*GPU_CONTEXT;
        ctx.reset();
        let evaluator = <G as Gate<F>>::Evaluator::new_from_parameters(params);
        let capture = GPUDataCapture::from_evaluator(evaluator);
        Description { capture, gate_type }
    }
}

pub(super) fn generate() {
    let descriptions = get_descriptions();
    assert!(descriptions
        .iter()
        .map(|d| &d.capture.evaluator_name)
        .all_unique());
    generate_cuda(&descriptions);
    generate_rust(&descriptions);
}

fn generate_cuda(descriptions: &[Description]) {
    const TEMPLATE_PATH: &str = "native/gates_template.cu";
    const RESULT_PATH: &str = "native/gates.cu";
    let mut code = String::new();
    let s = &mut code;
    new_line(s);
    for description in descriptions {
        let data = get_data(description);
        let capture = &description.capture;
        let name = data.name;
        s.push_str(format!("GATE_FUNCTION({name}) {{").as_str());
        new_line(s);
        indent(s, 1);
        s.push_str("GATE_INIT");
        new_line(s);
        indent(s, 1);
        s.push_str("GATE_REP {");
        new_line(s);
        match data.gate_type {
            Generic => {
                if let Some(index) = data.max_variable_index {
                    indent(s, 2);
                    let index = index + 1;
                    s.push_str(format!("GATE_VARS({index})").as_str());
                    new_line(s);
                }
                if let Some(index) = data.max_witness_index {
                    indent(s, 2);
                    let index = index + 1;
                    s.push_str(format!("GATE_WITS({index})").as_str());
                    new_line(s);
                }
                if let Some(index) = data.max_constant_index {
                    indent(s, 2);
                    let index = index + 1;
                    s.push_str(format!("GATE_CONS({index})").as_str());
                    new_line(s);
                }
                if let Some(index) = data.max_temp_index {
                    indent(s, 2);
                    let index = index + 1;
                    s.push_str(format!("GATE_TEMPS({index})").as_str());
                    new_line(s);
                }
                for (lhs, rhs) in &capture.relations {
                    indent(s, 2);
                    let lhs = get_index_string(&lhs.idx);
                    s.push_str(
                        match rhs {
                            Relation::Add(x, y) => {
                                format!(
                                    "GATE_ADD({}, {}, {lhs})",
                                    get_index_string(x),
                                    get_index_string(y)
                                )
                            }
                            Relation::Double(x) => {
                                format!("GATE_DBL({}, {lhs})", get_index_string(x))
                            }
                            Relation::Sub(x, y) => {
                                format!(
                                    "GATE_SUB({}, {}, {lhs})",
                                    get_index_string(x),
                                    get_index_string(y)
                                )
                            }
                            Relation::Negate(x) => {
                                format!("GATE_NEG({}, {lhs})", get_index_string(x))
                            }
                            Relation::Mul(x, y) => {
                                format!(
                                    "GATE_MUL({}, {}, {lhs})",
                                    get_index_string(x),
                                    get_index_string(y)
                                )
                            }
                            Relation::Square(x) => {
                                format!("GATE_SQR({}, {lhs})", get_index_string(x))
                            }
                            Relation::Inverse(x) => {
                                format!("GATE_INV({}, {lhs})", get_index_string(x))
                            }
                        }
                        .as_str(),
                    );
                    new_line(s);
                }
                for write in &capture.writes_per_repetition {
                    indent(s, 2);
                    s.push_str(format!("GATE_PUSH({})", get_index_string(write)).as_str());
                    new_line(s);
                }
            }
            GateType::Poseidon2(variables_count, witnesses_count) => {
                indent(s, 2);
                s.push_str(
                    format!("GATE_POSEIDON2({variables_count}, {witnesses_count})").as_str(),
                );
                new_line(s);
            }
            GateType::Poseidon2ExternalMatrix => {
                indent(s, 2);
                s.push_str("GATE_POSEIDON2_EXTERNAL_MATRIX");
                new_line(s);
            }
            GateType::Poseidon2InternalMatrix => {
                indent(s, 2);
                s.push_str("GATE_POSEIDON2_INTERNAL_MATRIX");
                new_line(s);
            }
        }
        indent(s, 1);
        s.push('}');
        new_line(s);
        s.push('}');
        new_line(s);
        new_line(s);
        s.push_str(format!("GATE_KERNEL({name})").as_str());
        new_line(s);
        new_line(s);
    }
    super::template::generate(&[("CODE", code)], TEMPLATE_PATH, RESULT_PATH);
}

fn generate_rust(descriptions: &[Description]) {
    const TEMPLATE_PATH: &str = "src/gates_data_template.rs";
    const RESULT_PATH: &str = "src/gates_data.rs";
    let mut hash_map = String::new();
    let mut bindings = String::new();
    let mut mappings = String::new();
    let h = &mut hash_map;
    let b = &mut bindings;
    let m = &mut mappings;
    new_line(b);
    for (id, description) in descriptions.iter().enumerate() {
        let data = get_data(description);
        let name = data.name;
        indent(h, 2);
        h.push_str(format!("(\"{name}\", {id}),").as_str());
        new_line(h);
        let kernel_name = format!("evaluate_{name}_kernel");
        b.push_str(format!("gate_eval_kernel!({kernel_name});").as_str());
        new_line(b);
        new_line(m);
        indent(m, 2);
        m.push_str(format!("{id} => GateData {{").as_str());
        new_line(m);
        push_data_param(m, "name", format!("\"{name}\""));
        push_data_param(m, "contributions_count", data.contributions_count);
        push_data_param_option(m, "max_variable_index", data.max_variable_index);
        push_data_param_option(m, "max_witness_index", data.max_witness_index);
        push_data_param_option(m, "max_constant_index", data.max_constant_index);
        push_data_param(m, "kernel", kernel_name);
        indent(m, 2);
        m.push_str("},");
    }
    super::template::generate(
        &[
            ("HASH_MAP", hash_map),
            ("BINDINGS", bindings),
            ("MAPPINGS", mappings),
        ],
        TEMPLATE_PATH,
        RESULT_PATH,
    );
}

fn push_data_param(target: &mut String, name: &str, value: impl Display) {
    indent(target, 3);
    target.push_str(format!("{name}: {value},").as_str());
    new_line(target);
}

fn push_data_param_option(target: &mut String, name: &str, value: Option<impl Display>) {
    push_data_param(
        target,
        name,
        value.map_or(String::from("None"), |x| format!("Some({x})")),
    );
}

#[derive(Debug)]
struct GateData {
    gate_type: GateType,
    name: String,
    contributions_count: u32,
    max_variable_index: Option<u32>,
    max_witness_index: Option<u32>,
    max_constant_index: Option<u32>,
    max_temp_index: Option<u32>,
}

fn get_data(description: &Description) -> GateData {
    let gate_type = description.gate_type;
    let capture = &description.capture;
    let name = capture.evaluator_name.clone();
    let indexes = get_indexes(&capture.relations);
    let max_variable_index = indexes
        .iter()
        .filter_map(|x| match x {
            Index::VariablePoly(y) => Some(*y as u32),
            _ => None,
        })
        .max();
    let max_witness_index = indexes
        .iter()
        .filter_map(|x| match x {
            Index::WitnessPoly(y) => Some(*y as u32),
            _ => None,
        })
        .max();
    let max_constant_index = indexes
        .iter()
        .filter_map(|x| match x {
            Index::ConstantPoly(y) => Some(*y as u32),
            _ => None,
        })
        .max();
    let max_temp_index = indexes
        .iter()
        .filter_map(|x| match x {
            Index::TemporaryValue(y) => Some(*y as u32),
            _ => None,
        })
        .max();
    let contributions_count = capture.writes_per_repetition.len() as u32;
    GateData {
        gate_type,
        name,
        contributions_count,
        max_variable_index,
        max_witness_index,
        max_constant_index,
        max_temp_index,
    }
}

fn get_index_string(idx: &Index<F>) -> String {
    match idx {
        Index::VariablePoly(i) => {
            format!("v[{i}]")
        }
        Index::WitnessPoly(i) => {
            format!("w[{i}]")
        }
        Index::ConstantPoly(i) => {
            format!("c[{i}]")
        }
        Index::TemporaryValue(i) => {
            format!("t[{i}]")
        }
        Index::ConstantValue(value) => {
            format!("GATE_VAL({:#018x})", value.as_u64())
        }
    }
}

fn indent(s: &mut String, count: u32) {
    for _ in 0..count {
        s.push_str("    ");
    }
}

fn new_line(s: &mut String) {
    s.push('\n');
}

fn get_indexes(relations: &Vec<(GpuSynthesizerFieldLike<F>, Relation<F>)>) -> HashSet<Index<F>> {
    let mut indexes = HashSet::new();
    for (lhs, rhs) in relations {
        indexes.insert(lhs.idx);
        match rhs {
            Relation::Add(x, y) | Relation::Sub(x, y) | Relation::Mul(x, y) => {
                indexes.insert(*x);
                indexes.insert(*y);
            }
            Relation::Double(x)
            | Relation::Negate(x)
            | Relation::Square(x)
            | Relation::Inverse(x) => {
                indexes.insert(*x);
            }
        };
    }
    indexes
}

fn get_descriptions() -> Vec<Description> {
    vec![
        Description::new::<BooleanConstraintGate>((), Generic),
        Description::new::<ConditionalSwapGate<1>>((), Generic),
        Description::new::<ConditionalSwapGate<4>>((), Generic),
        Description::new::<ConstantsAllocatorGate<F>>((), Generic),
        Description::new::<DotProductGate<4>>((), Generic),
        Description::new::<FmaGateInBaseFieldWithoutConstant<F>>((), Generic),
        Description::new::<FmaGateInExtensionWithoutConstant<F, GoldilocksExt2>>((), Generic),
        Description::new::<MatrixMultiplicationGate<F, 12, Poseidon2GoldilocksExternalMatrix>>(
            (),
            GateType::Poseidon2ExternalMatrix,
        ),
        Description::new::<MatrixMultiplicationGate<F, 12, Poseidon2GoldilocksInnerMatrix>>(
            (),
            GateType::Poseidon2InternalMatrix,
        ),
        Description::new::<ParallelSelectionGate<4>>((), Generic),
        Description::new::<Poseidon2FlattenedGate<F, 8, 12, 4, Poseidon2Goldilocks>>(
            (130, 0),
            GateType::Poseidon2(130, 0),
        ),
        Description::new::<Poseidon2FlattenedGate<F, 8, 12, 4, Poseidon2Goldilocks>>(
            (100, 30),
            GateType::Poseidon2(100, 30),
        ),
        Description::new::<QuadraticCombinationGate<4>>((), Generic),
        Description::new::<ReductionByPowersGate<F, 4>>((), Generic),
        Description::new::<ReductionGate<F, 4>>((), Generic),
        Description::new::<SelectionGate>((), Generic),
        Description::new::<SimpleNonlinearityGate<F, 7>>((), Generic),
        Description::new::<U32AddGate>((), Generic),
        Description::new::<U8x4FMAGate>((), Generic),
        Description::new::<U32SubGate>((), Generic),
        Description::new::<U32TriAddCarryAsChunkGate>((), Generic),
        Description::new::<UIntXAddGate<32>>((), Generic),
        Description::new::<ZeroCheckGate>(false, Generic),
        Description::new::<ZeroCheckGate>(true, Generic),
    ]
}
