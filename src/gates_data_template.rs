use std::collections::HashMap;

use lazy_static::lazy_static;

lazy_static! {
    static ref HASH_MAP: HashMap<&'static str, u32> = [
%HASH_MAP%    ]
    .iter()
    .copied()
    .collect();
}

extern "C" {%BINDINGS%}

fn get_gate_data(id: u32) -> GateData {
    match id {%MAPPINGS%
        _ => panic!("unknown gate id {id}"),
    }
}
