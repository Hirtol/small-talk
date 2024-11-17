//! All content related to the back-end systems such as voice generation

use std::collections::HashMap;
pub use data::*;
use std::sync::Arc;
use serde::{Deserialize, Serialize};
use crate::system::tts_backends::alltalk::AllTalkTTS;

mod tts_backends;
pub mod data;
pub mod session;

pub type VoiceSystemHandle = Arc<VoiceSystem>;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Hash, Ord, PartialOrd, Eq, PartialEq)]
pub enum TtsModel {
    F5,
    Xtts,
}

pub struct VoiceSystem {
    backends: HashMap<TtsModel, AllTalkTTS>,
}

impl VoiceSystem {
    pub fn new() -> Self {
        Self {
            backends: Default::default(),
        }
    }
}

pub struct QuickRelease<T> {
    inner: Option<T>
}