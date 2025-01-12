use burn::prelude::{Backend, Config, Module};

pub type CpuBackend = burn::backend::NdArray<f32, i32>;
pub type GpuBackend = burn::backend::Wgpu<f32, i32>;

pub use burn;

pub mod embeddings;
pub mod emotion_classifier;
pub mod stt;

