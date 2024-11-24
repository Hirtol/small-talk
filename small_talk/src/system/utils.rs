use rand::{thread_rng, Rng};
use rand::distributions::Alphanumeric;

/// Generate a random file name 
#[inline]
pub fn random_file_name(length: usize, extension: Option<&str>) -> String {
    let name: String = thread_rng()
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