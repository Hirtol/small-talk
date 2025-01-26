use std::sync::{Arc, Mutex};
use eyre::Context;
use st_ml::CpuBackend;
use crate::{config::TtsSystemConfig, error::EmotionError};
pub use st_ml::emotion_classifier::{BasicEmotion, BasicEmotionClassifier};

#[derive(Clone)]
pub struct EmotionBackend {
    model: Arc<Mutex<BasicEmotionClassifier<CpuBackend>>>,
}

impl EmotionBackend {
    pub fn new(config: &TtsSystemConfig) -> Result<EmotionBackend, EmotionError> {
        let device = st_ml::burn::backend::ndarray::NdArrayDevice::default();
        let classifier =
            BasicEmotionClassifier::new(&config.emotion_classifier_model, &config.bert_embeddings_model, device)?;
        Ok(Self { model: Arc::new(Mutex::new(classifier)) })
    }

    /// Try to (batch) classify all the given texts, returning a [Vec] containing the emotions for the texts in-order.
    pub fn classify_emotion(&mut self, texts: impl IntoIterator<Item = impl AsRef<str>>) -> Result<Vec<BasicEmotion>, EmotionError> {
        let mut lock = self.model.lock().expect("Poisoned");
        Ok(lock.infer(texts)?)
    }
}
