//! All content related to the back-end systems such as voice generation

use std::sync::Arc;

pub type VoiceSystemHandle = Arc<VoiceSystem>;

#[derive(Debug)]
pub struct VoiceSystem {
    
}

impl VoiceSystem {
    pub fn new() -> Self {
        Self {
            
        }
    }
}

pub struct QuickRelease<T> {
    inner: Option<T>
}