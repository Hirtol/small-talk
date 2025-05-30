use crate::{
    embeddings::LLamaEmbedder,
    emotion_classifier::{
        data::EmotionBatcher,
        model::EmotionModel,
        training::{TrainingConfig},
    },
};
use burn::{
    backend::NdArray,
    prelude::{Backend, Config, Module},
    record::{CompactRecorder, Recorder},
};
use error_set::error_set;
use llama_cpp_2::{context::params::LlamaContextParams, model::params::LlamaModelParams};
use std::{
    fmt::Debug,
    path::{Path, PathBuf},
};

pub mod data;
pub mod model;
pub mod training;

error_set! {
    LoadError = {
        #[display("Could not find the model at {path:?}")]
        ModelNotFound {
            path: PathBuf
        },
        BurnConfig(burn::config::ConfigError),
        Eyre(eyre::Error)
    };
    OutOfRangeError = {
        NoEmotionMapped,
    };
}

pub struct BasicEmotionClassifier<B: Backend = NdArray> {
    /// Classifier model, simple linear layer on top of the headings provided by `llama_embedder`
    model: EmotionModel<B>,
    /// BERT-based model which will generate snippet embeddings. We use Llama.cpp as its CPU inference speed is
    /// literally 10 to 100 times faster than implementing it in Rust (irrespective of frameworks atm, they all suck for CPU inference).
    llama_embedder: LLamaEmbedder,
    batcher: EmotionBatcher<B>,
    device: B::Device,
}

impl<B: Backend> BasicEmotionClassifier<B> {
    /// Create a new emotion classifier
    #[tracing::instrument]
    pub fn new(
        classifier_path: impl AsRef<Path> + Debug,
        embedder_path: impl AsRef<Path> + Debug,
        device: B::Device,
    ) -> Result<Self, LoadError> {
        tracing::trace!("Loading emotion classifier");
        let classifier = classifier_path.as_ref();
        let config = TrainingConfig::load(classifier.join("config.json"))?;
        let record = CompactRecorder::new()
            .load(classifier.join("model"), &device)
            .expect("Trained model should exist");

        let model = config.model.init::<B>(&device).load_record(record);

        tracing::trace!("Loading BERT embedding model");
        let model_params = LlamaModelParams::default().with_n_gpu_layers(0);
        let ctx_params = LlamaContextParams::default()
            .with_n_threads(16)
            .with_n_threads_batch(16)
            .with_n_ctx(None) // Load from model
            .with_n_batch(512)
            .with_embeddings(true);
        let llama = LLamaEmbedder::new(embedder_path, model_params, ctx_params, None)?;

        Ok(Self {
            model,
            llama_embedder: llama,
            batcher: EmotionBatcher::new(device.clone()),
            device,
        })
    }

    /// Infer the [BasicEmotion] of each text snippet provided in `texts`.
    ///
    /// # Arguments
    /// * `texts` - An ordered iterator, the first item in the result will match with the first text snippet in the iterator.
    #[tracing::instrument(skip_all)]
    pub fn infer(&mut self, texts: impl IntoIterator<Item = impl AsRef<str>>) -> Result<Vec<BasicEmotion>, LoadError> {
        let embeddings = self.llama_embedder.embed(texts, false, true)?;
        let embedding_tensor = model::embed_to_tensor(embeddings, &self.device);

        let output = self.model.forward(embedding_tensor);
        let classes = output.argmax(1).flatten::<1>(0, 1).into_data();
        let classes_indexes: &[i32] = classes.as_slice().expect("Invalid data cast");
        Ok(classes_indexes
            .iter()
            .copied()
            .flat_map(BasicEmotion::try_from)
            .collect())
    }
}

pub const BASIC_EMOTIONS: [&str; 8] = [
    "neutral",
    "non-neutral",
    "joy",
    "surprise",
    "anger",
    "sadness",
    "disgust",
    "fear",
];

#[derive(Debug, Copy, Clone, PartialEq, Ord, PartialOrd, Eq, Default, Hash, serde::Serialize, serde::Deserialize)]
pub enum BasicEmotion {
    #[default]
    Neutral = 0,
    NonNeutral = 1,
    Joy = 2,
    Surprise = 3,
    Anger = 4,
    Sadness = 5,
    Disgust = 6,
    Fear = 7,
}

impl BasicEmotion {
    /// Return a constant array with a preferred order for each [BasicEmotion].
    ///
    /// If we're trying to find something to match a given [BasicEmotion] this indicates a possible order which makes the most sense.
    pub const fn to_preference_order(&self) -> [Self; 8] {
        use BasicEmotion::*;
        match self {
            Neutral => [Neutral, NonNeutral, Surprise, Joy, Sadness, Anger, Disgust, Fear],
            NonNeutral => [NonNeutral, Neutral, Surprise, Joy, Sadness, Anger, Disgust, Fear],
            Joy => [Joy, Surprise, Neutral, NonNeutral, Sadness, Anger, Disgust, Fear],
            Surprise => [Surprise, Neutral, Joy, NonNeutral, Sadness, Anger, Disgust, Fear],
            Anger => [Anger, Neutral, Sadness, Disgust, Fear, Joy, Surprise, NonNeutral],
            Sadness => [Sadness, Neutral, Anger, Disgust, Fear, Joy, Surprise, NonNeutral],
            Disgust => [Disgust, Neutral, Anger, Sadness, Fear, Joy, Surprise, NonNeutral],
            Fear => [Fear, Neutral, Sadness, Disgust, Anger, Joy, Surprise, NonNeutral],
        }
    }

    pub fn matches_file(&self, file_name: &str) -> bool {
        file_name.to_lowercase().contains(BASIC_EMOTIONS[*self as usize])
    }

    pub fn from_file_name(file_name: &str) -> Option<BasicEmotion> {
        let lower_case = file_name.to_lowercase();
        for (i, emotion) in BASIC_EMOTIONS.iter().enumerate() {
            if lower_case.contains(emotion) {
                return BasicEmotion::try_from(i as i32).ok();
            }
        }

        None
    }
}

impl TryFrom<i32> for BasicEmotion {
    type Error = OutOfRangeError;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(BasicEmotion::Neutral),
            1 => Ok(BasicEmotion::NonNeutral),
            2 => Ok(BasicEmotion::Joy),
            3 => Ok(BasicEmotion::Surprise),
            4 => Ok(BasicEmotion::Anger),
            5 => Ok(BasicEmotion::Sadness),
            6 => Ok(BasicEmotion::Disgust),
            7 => Ok(BasicEmotion::Fear),
            _ => Err(OutOfRangeError::NoEmotionMapped),
        }
    }
}
