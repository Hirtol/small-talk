use std::cell::RefCell;
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use burn::data::dataloader::batcher::Batcher;
use burn::data::dataloader::DataLoaderBuilder;
use burn::data::dataset::{Dataset, SqliteDataset};
use burn::optim::AdamConfig;
use burn::prelude::{Backend, Config, ElementConversion, Int, Module, Tensor, TensorData};
use burn::record::CompactRecorder;
use burn::tensor::backend::AutodiffBackend;
use burn::train::{ClassificationOutput, LearnerBuilder, TrainOutput, TrainStep, ValidStep};
use burn::train::metric::{AccuracyMetric, LossMetric};
use serde::{Deserialize, Serialize};
use crate::embeddings::LLamaEmbedder;
use crate::{EmotionModel, EmotionModelConfig};

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

#[derive(Clone)]
pub struct EmotionBatcher<B: Backend> {
    device: B::Device,
    llama: LlamaTrain,
}

impl<B: Backend> EmotionBatcher<B> {
    pub fn new(embedder: LlamaTrain, device: B::Device) -> Self {
        Self {
            device,
            llama: embedder,
        }
    }
}

impl<B: Backend> Batcher<GoEmotionItem, EmotionBatch<B>> for EmotionBatcher<B> {
    fn batch(&self, items: Vec<GoEmotionItem>) -> EmotionBatch<B> {
        let embeddings = self
            .llama
            .lock()
            .unwrap()
            .embed(items.iter().map(|i| &i.text), false, true)
            .unwrap();
        let emb_shape = [embeddings[0].len()];
        
        let embeddings2 = embeddings
            .into_iter()
            .map(|emb| TensorData::new(emb, emb_shape).convert::<B::FloatElem>())
            .map(|data| Tensor::<B, 1>::from_data(data, &self.device))
            .map(|tens| tens.reshape([1, emb_shape[0]]))
            .collect();
        
        let targets = items
            .iter()
            .map(|item| Tensor::<B, 1, Int>::from_data([item.labels[0].elem::<B::IntElem>()], &self.device))
            .collect();
        
        let embeddings = Tensor::cat(embeddings2, 0).to_device(&self.device);
        let targets = Tensor::cat(targets, 0).to_device(&self.device);
        
        EmotionBatch { embeddings, targets }
    }
}

#[derive(Clone, Debug)]
pub struct EmotionBatch<B: Backend> {
    pub embeddings: Tensor<B, 2>,
    pub targets: Tensor<B, 1, Int>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoEmotionItem {
    pub text: String,
    pub labels: Vec<u32>,
    pub id: String,
}

impl<B: AutodiffBackend> TrainStep<EmotionBatch<B>, ClassificationOutput<B>> for EmotionModel<B> {
    fn step(&self, batch: EmotionBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(batch.embeddings, batch.targets);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<EmotionBatch<B>, ClassificationOutput<B>> for EmotionModel<B> {
    fn step(&self, batch: EmotionBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(batch.embeddings, batch.targets)
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

pub fn train<B: AutodiffBackend>(artifact_dir: &str, train: SqliteDataset<GoEmotionItem>, test: SqliteDataset<GoEmotionItem>, llama_embedder: LLamaEmbedder, config: TrainingConfig, device: B::Device) {
    create_artifact_dir(artifact_dir);
    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Config should be saved successfully");

    B::seed(config.seed);

    let llama_train = LlamaTrain::new(Mutex::new(NullLock(llama_embedder)));
    let batcher_train = EmotionBatcher::<B>::new(llama_train.clone(), device.clone());
    let batcher_valid = EmotionBatcher::<B::InnerBackend>::new(llama_train, device.clone());

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