use std::env::var;
use std::fs;
use std::path::Path;

const PREFIX: char = '%';
const SUFFIX: char = '%';

pub(crate) fn generate(replacements: &[(&str, String)], template_path: &str, result_path: &str) {
    let mut text = fs::read_to_string(template_path).unwrap();
    for (key, value) in replacements {
        let mut from = String::from(PREFIX);
        from.push_str(key);
        from.push(SUFFIX);
        text = text.replace(&from, value);
    }
    let out_dir = var("OUT_DIR").unwrap();
    let result_path = Path::new(&out_dir).join(result_path);
    let current = fs::read_to_string(&result_path).unwrap_or_default();
    if !text.eq(&current) {
        fs::write(&result_path, text).unwrap();
    }
}
