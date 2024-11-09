use std::cell::RefCell;
use std::collections::HashMap;
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use burn::data::dataloader::batcher::Batcher;
use burn::data::dataloader::DataLoaderBuilder;
use burn::data::dataset::{Dataset, InMemDataset, SqliteDataset};
use burn::optim::AdamConfig;
use burn::prelude::{Backend, Config, ElementConversion, Int, Module, Tensor, TensorData};
use burn::record::CompactRecorder;
use burn::tensor::backend::AutodiffBackend;
use burn::train::{ClassificationOutput, LearnerBuilder, TrainOutput, TrainStep, ValidStep};
use burn::train::metric::{AccuracyMetric, LossMetric};
use eyre::ContextCompat;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::model::params::LlamaModelParams;
use serde::{Deserialize, Serialize};
use crate::embeddings::LLamaEmbedder;
use crate::emotion_classifier::data::{EmotionBatcher, EmotionItem, EmotionTrainingBatch};
use crate::emotion_classifier::model::{EmotionModel, EmotionModelConfig};

pub mod mapper;

/// Horrifically unsafe, mainly here to get around the fact that all `frida_gum` types are `!Send + !Sync` due to the raw
/// pointers embedded in their structs.
pub struct NullLock<T>(pub T);

impl<T> std::ops::Deref for NullLock<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl<T> std::ops::DerefMut for NullLock<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

unsafe impl<T> Sync for NullLock<T> {}
unsafe impl<T> Send for NullLock<T> {}

pub type LlamaTrain = Arc<Mutex<NullLock<LLamaEmbedder>>>;


pub struct LLamaTrainEmbedder {
    embedder: LLamaEmbedder,
    data: HashMap<String, Vec<f32>>,
    cache_path: PathBuf,
}

impl LLamaTrainEmbedder {
    pub fn new(embedding_model: impl AsRef<Path>, ctx: LlamaContextParams, cache_path: impl Into<PathBuf>) -> eyre::Result<Self> {
        let model_params = LlamaModelParams::default();

        let emb =  LLamaEmbedder::new(embedding_model, model_params, ctx, None)?;
        let cache_path = cache_path.into();
        
        Ok(Self {
            embedder: emb,
            data: Self::load_or_default(&cache_path),
            cache_path,
        })
    }
    
    /// Either create new embeddings or query the cache.
    pub fn embed(&mut self, texts: impl IntoIterator<Item = impl AsRef<str>>) -> eyre::Result<Vec<Vec<f32>>> {
        let mut output = Vec::new();
        for text in texts {
            let text = text.as_ref();
            if let Some(data) = self.data.get(text) {
                output.push(data.clone())
            } else {
                let mut embeddings = self.embedder.embed([text], false, true)?;
                self.data.insert(text.into(), embeddings.pop().unwrap());
                output.push(self.data.get(text).unwrap().clone());
            }
        }

        Ok(output)
    }
    
    fn load_or_default(cache: impl AsRef<Path>) -> HashMap<String, Vec<f32>> {
        fn fallible(cache: &Path) -> eyre::Result<HashMap<String, Vec<f32>>> {
            Ok(serde_json::from_slice(&std::fs::read(cache)?)?)
        }
        
        fallible(cache.as_ref()).unwrap_or_default()
    }
    
    pub fn save_cache(&self) -> eyre::Result<()> {
        std::fs::create_dir_all(self.cache_path.parent().context("No parent")?)?;
        let write = BufWriter::new(std::fs::OpenOptions::new().write(true).create(true).open(&self.cache_path)?);
        serde_json::to_writer(write, &self.data)?;
        
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FriendsEmotionItem {
    pub text: String,
    pub label: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoEmotionItem {
    pub text: String,
    pub labels: Vec<u32>,
    pub id: String,
}

impl<B: AutodiffBackend> TrainStep<EmotionTrainingBatch<B>, ClassificationOutput<B>> for EmotionModel<B> {
    fn step(&self, batch: EmotionTrainingBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(batch.embeddings, batch.labels);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<EmotionTrainingBatch<B>, ClassificationOutput<B>> for EmotionModel<B> {
    fn step(&self, batch: EmotionTrainingBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(batch.embeddings, batch.labels)
    }
}

#[derive(Config)]
pub struct TrainingConfig {
    pub model: EmotionModelConfig,
    pub optimizer: AdamConfig,
    #[config(default = 10)]
    pub num_epochs: usize,
    #[config(default = 64)]
    pub batch_size: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1.0e-4)]
    pub learning_rate: f64,
}

fn create_artifact_dir(artifact_dir: &str) {
    // Remove existing artifacts before to get an accurate learner summary
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

pub fn train<B: AutodiffBackend>(artifact_dir: &str, train: impl Dataset<EmotionItem> + 'static, test: impl Dataset<EmotionItem> + 'static, config: TrainingConfig, device: B::Device) {
    create_artifact_dir(artifact_dir);
    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Config should be saved successfully");

    B::seed(config.seed);
    
    tracing::info!("Starting embeddings pre-calculation, this may take a while");
    
    
    let batcher_train = EmotionBatcher::<B>::new(device.clone());
    let batcher_valid = EmotionBatcher::<B::InnerBackend>::new(device.clone());

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(train);

    let dataloader_test = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(test);

    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .summary()
        .build(
            config.model.init::<B>(&device),
            config.optimizer.init(),
            config.learning_rate,
        );

    let model_trained = learner.fit(dataloader_train, dataloader_test);

    model_trained
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Trained model should be saved successfully");
}