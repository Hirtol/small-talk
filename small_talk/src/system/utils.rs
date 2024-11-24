use rand::{thread_rng, Rng};
use rand::distributions::Alphanumeric;

/// Generate a random file name 
#[inline]
pub fn random_file_name(length: usize, extension: &str) -> String {
    let name: String = thread_rng()
        .sample_iter(&Alphanumeric)
        .take(length)
        .map(char::from)
        .collect();
    
    format!("{}.{}", name, extension)
}