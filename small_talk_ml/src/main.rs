use burn::{
    backend::Wgpu,
    data::{
        dataloader::batcher::Batcher,
        dataset::{Dataset, HuggingfaceDatasetLoader, SqliteDataset},
    },
    prelude::{Backend, ElementConversion, Int, TensorData},
    tensor::Tensor,
};
use llama_cpp_2::model::params::LlamaModelParams;
use serde::{Deserialize, Serialize};
use small_talk_ml::{embeddings::LLamaEmbedder, EmotionModelConfig};
use std::cell::RefCell;
use std::cmp::Reverse;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use burn::backend::Autodiff;
use burn::optim::AdamConfig;
use burn::prelude::{Config, Module};
use burn::record::{CompactRecorder, Recorder};
use burn::tensor::activation::softmax;
use burn::tensor::cast::ToElement;
use itertools::Itertools;
use llama_cpp_2::context::params::LlamaContextParams;
use small_talk_ml::training::{GoEmotionItem, TrainingConfig};

const CLASSES: [&str; 28] = [
    "admiration",
    "amusement",
    "anger",
    "annoyance",
    "approval",
    "caring",
    "confusion",
    "curiosity",
    "desire",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "excitement",
    "fear",
    "gratitude",
    "grief",
    "joy",
    "love",
    "nervousness",
    "optimism",
    "pride",
    "realization",
    "relief",
    "remorse",
    "sadness",
    "surprise",
    "neutral",
];

type Back = Wgpu<f32, i32>;
type MyAuto = Autodiff<Back>;

fn main() -> eyre::Result<()> {
    let device = Default::default();
    let model_params = LlamaModelParams::default().with_n_gpu_layers(0);
    let ctx_params = LlamaContextParams::default()
        .with_n_threads(16)
        .with_n_threads_batch(16)
        .with_n_ctx(None) // Load from model
        .with_n_batch(512)
        .with_embeddings(true);
    let mut llama = LLamaEmbedder::new(
        "G:\\ML Models\\Embedding Models\\bge-small-en-v1.5\\ggml-model-Q4_k.gguf",
        model_params,
        ctx_params,
        None,
    )?;
    let model = EmotionModelConfig::new(384, 28).init::<Back>(&device);
    let model_cfg = EmotionModelConfig::new(384, 28);

    let train: SqliteDataset<GoEmotionItem> =
        HuggingfaceDatasetLoader::new("google-research-datasets/go_emotions").dataset("train")?;
    let test: SqliteDataset<GoEmotionItem> =
        HuggingfaceDatasetLoader::new("google-research-datasets/go_emotions").dataset("test")?;

    println!("Model: {model:?}");
    println!("Data: {:?}", train.get(0));

    let now = Instant::now();

    let embeddings = llama.embed(["\"Johny, why are you going over there? Johny?!\"", "Oh my clowns, great... I'm going to fucking kill you"], false, true)?;
    let emb_shape = [embeddings[0].len()];

    let mut embeddings2 = embeddings
        .into_iter()
        .map(|emb| TensorData::new(emb, emb_shape).convert::<f32>())
        .map(|data| Tensor::<Back, 1>::from_data(data, &device))
        .map(|tens| tens.reshape([1, emb_shape[0]]))
        .collect::<Vec<_>>();
    let embeddings = Tensor::cat(embeddings2, 0).to_device(&device);
    infer("./artifacts", device, embeddings);
    // let logits = model.forward(embeddings2.pop().unwrap());
    // println!("Logits: {logits:?}");

    // small_talk_ml::training::train::<MyAuto>("./artifacts", train, test, llama, TrainingConfig::new(model_cfg, AdamConfig::new()), device);

    println!("Training took: {:?}", now.elapsed());
    Ok(())
}

pub fn infer<B: Backend>(artifact_dir: &str, device: B::Device, embedding: Tensor<B, 2>) {
    let config = TrainingConfig::load(format!("{artifact_dir}/config.json"))
        .expect("Config should exist for the model");
    let record = CompactRecorder::new()
        .load(format!("{artifact_dir}/model").into(), &device)
        .expect("Trained model should exist");

    let model = config.model.init::<B>(&device).load_record(record);

    // let label = item.label;
    // let batcher = MnistBatcher::new(device);
    // let batch = batcher.batch(vec![item]);
    let output = model.forward(embedding);
    
    let probabilities = softmax(output.clone(), 1);
    for slice in probabilities.iter_dim(0) {
        let tensor_data = slice.into_data();
        let data = tensor_data.as_slice::<f32>().unwrap();
        let top_5 = data.iter().enumerate().map(|(i, prob)| (CLASSES[i], prob)).sorted_by(|a, b| b.1.partial_cmp(a.1).unwrap()).take(5).collect_vec();
        
        println!("Slice: {top_5:#?}");
    }
    
    println!("Output: {}", output.clone().argmax(1).flatten::<1>(0, 1));
    println!("Output Soft: {}", softmax(output.clone(), 1).into_data());
    let predicted = output.argmax(1).flatten::<1>(0, 1).into_scalar();

    println!("Predicted {} = {}", predicted, CLASSES[predicted.to_usize()]);
}
