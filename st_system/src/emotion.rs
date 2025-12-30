use std::path::{Path, PathBuf};
use crate::{config::TtsSystemConfig, error::EmotionError};
use eyre::OptionExt;
pub use st_ml::emotion_classifier::{BasicEmotion, BasicEmotionClassifier};
use st_ml::CpuBackend;
use std::sync::{Arc, Mutex};

#[derive(Clone)]
pub struct EmotionCoordinator {
    classifier: Option<EmotionBackend>,
}

impl EmotionCoordinator {
    pub fn new(backend: Option<EmotionBackend>) -> Self {
        Self { classifier: backend }
    }

    /// Try to (batch) classify all the given texts, returning a [Vec] containing the emotions for the texts in-order.
    ///
    /// Will block until everything is classified.
    pub fn classify_emotion(
        &mut self,
        texts: impl IntoIterator<Item = impl AsRef<str>>,
    ) -> Result<Vec<BasicEmotion>, EmotionError> {
        self.classifier
            .as_mut()
            .ok_or_eyre("The emotion classifier is mandatory in order to classify emotions")?
            .classify_emotion(texts)
    }
}

#[derive(Clone)]
pub struct EmotionBackend {
    model: Arc<Mutex<BasicEmotionClassifier<CpuBackend>>>,
}

impl EmotionBackend {
    pub fn new(emotion_classifier_model: &Path, bert_embeddings_model: &Path) -> Result<EmotionBackend, EmotionError> {
        let device = st_ml::burn::backend::ndarray::NdArrayDevice::default();
        let classifier =
            // BasicEmotionClassifier::new(&config.emotion_classifier_model, &config.bert_embeddings_model, device)?;
            BasicEmotionClassifier::new(emotion_classifier_model, bert_embeddings_model, device)?;
        Ok(Self {
            model: Arc::new(Mutex::new(classifier)),
        })
    }

    /// Try to (batch) classify all the given texts, returning a [Vec] containing the emotions for the texts in-order.
    ///
    /// Will block until everything is classified.
    pub fn classify_emotion(
        &mut self,
        texts: impl IntoIterator<Item = impl AsRef<str>>,
    ) -> Result<Vec<BasicEmotion>, EmotionError> {
        let mut lock = self.model.lock().expect("Poisoned");
        Ok(lock.infer(texts)?)
    }
}
