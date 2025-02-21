use rand::{Rng};
use rand::distr::Alphanumeric;

/// Generate a random file name 
#[inline]
pub fn random_file_name(length: usize, extension: Option<&str>) -> String {
    let name: String = rand::rng()
        .sample_iter(&Alphanumeric)
        .take(length)
        .map(char::from)
        .collect();
    if let Some(ext) = extension {
        format!("{}.{}", name, ext)
    } else {
        name
    }
}