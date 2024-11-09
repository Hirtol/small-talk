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
use small_talk_ml::{embeddings::LLamaEmbedder};
use std::cell::RefCell;
use std::cmp::Reverse;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use burn::backend::Autodiff;
use burn::backend::wgpu::WgpuDevice;
use burn::data::dataset::InMemDataset;
use burn::data::dataset::transform::{ComposedDataset, MapperDataset};
use burn::optim::AdamConfig;
use burn::prelude::{Config, Module};
use burn::record::{CompactRecorder, Recorder};
use burn::tensor::activation::softmax;
use burn::tensor::cast::ToElement;
use itertools::Itertools;
use llama_cpp_2::context::params::LlamaContextParams;
use small_talk_ml::emotion_classifier::data::EmotionItem;
use small_talk_ml::emotion_classifier::model::EmotionModelConfig;
use small_talk_ml::emotion_classifier::training;
use small_talk_ml::emotion_classifier::training::{FriendsEmotionItem, GoEmotionItem, LLamaTrainEmbedder, TrainingConfig};

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

const NEW_CLASSES: [&str; 8] = [
    "neutral",
    "non-neutral",
    "joy",
    "surprise",
    "anger",
    "sadness",
    "disgust",
    "fear"
];

type Back = Wgpu<f32, i32>;
type MyAuto = Autodiff<Back>;

fn main() -> eyre::Result<()> {
    train_setup()
}

pub fn infer_setup() -> eyre::Result<()> {
    let device = WgpuDevice::default();
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

    let now = Instant::now();

    let embeddings = llama.embed(["\"Johny, why are you going over there? Johny?!\"", "Oh my clowns, great... I'm going to fucking kill you", "You make me complete, happy, and whole!", "And now.. he's dead. And I'm left with nothing."], false, true)?;
    infer::<Back>("./artifacts_friends", &device, LLamaTrainEmbedder::embed_to_tensor(embeddings, &device));

    println!("Training took: {:?}", now.elapsed());
    Ok(())
}

pub fn train_setup() -> eyre::Result<()> {
    let device = Default::default();
    let ctx_params = LlamaContextParams::default()
        .with_n_threads(16)
        .with_n_threads_batch(16)
        .with_n_ctx(None) // Load from model
        .with_n_batch(512)
        .with_embeddings(true);

    let mut llama_cache = LLamaTrainEmbedder::new("G:\\ML Models\\Embedding Models\\bge-small-en-v1.5\\ggml-model-Q4_k.gguf", ctx_params, "./training/embedding_cache.json")?;

    let train: SqliteDataset<GoEmotionItem> =
        HuggingfaceDatasetLoader::new("google-research-datasets/go_emotions").dataset("train")?;
    let test: SqliteDataset<GoEmotionItem> =
        HuggingfaceDatasetLoader::new("google-research-datasets/go_emotions").dataset("test")?;

    let train_go_dataset = transform_go_item_dataset(&mut llama_cache, train)?;
    let test_go_dataset = transform_go_item_dataset(&mut llama_cache, test)?;

    let train_friends: SqliteDataset<FriendsEmotionItem> = HuggingfaceDatasetLoader::new("michellejieli/friends_dataset").dataset("train")?;
    let train_friend_dataset = transform_friend_dataset(&mut llama_cache, train_friends)?;

    let merged_train_dataset = ComposedDataset::new(vec![train_go_dataset, train_friend_dataset]);

    let model_cfg = EmotionModelConfig::new(384, 8);
    
    training::train::<MyAuto>("./artifacts_friends", merged_train_dataset, test_go_dataset, TrainingConfig::new(model_cfg, AdamConfig::new()), device);
    
    Ok(())
}

pub fn infer<B: Backend>(artifact_dir: &str, device: &B::Device, embedding: Tensor<B, 2>) {
    let config = TrainingConfig::load(format!("{artifact_dir}/config.json"))
        .expect("Config should exist for the model");
    let record = CompactRecorder::new()
        .load(format!("{artifact_dir}/model").into(), device)
        .expect("Trained model should exist");

    let model = config.model.init::<B>(device).load_record(record);
    
    let output = model.forward(embedding);

    let probabilities = softmax(output.clone(), 1);
    for slice in probabilities.iter_dim(0) {
        let tensor_data = slice.into_data();
        let data = tensor_data.as_slice::<f32>().unwrap();
        let top_5 = data.iter().enumerate().map(|(i, prob)| (NEW_CLASSES[i], prob)).sorted_by(|a, b| b.1.partial_cmp(a.1).unwrap()).take(5).collect_vec();

        println!("Slice: {top_5:#?}");
    }
}

#[tracing::instrument(skip_all)]
fn transform_go_item_dataset(cache: &mut LLamaTrainEmbedder, dataset: SqliteDataset<GoEmotionItem>) -> eyre::Result<InMemDataset<EmotionItem>> {
    let emotion_map = get_emotion_map();
    let embeddings = cache.embed(dataset.iter().map(|v| v.text))?;
    let test_dataset_vec = dataset.iter().zip(embeddings).map(|(item, embedding)| {
        EmotionItem {
            text_embedding: embedding,
            label: *emotion_map.get(CLASSES[item.labels[0] as usize]).unwrap(),
        }
    }).collect();
    let new_dataset = InMemDataset::new(test_dataset_vec);

    cache.save_cache()?;
    Ok(new_dataset)
}

#[tracing::instrument(skip_all)]
fn transform_friend_dataset(cache: &mut LLamaTrainEmbedder, dataset: SqliteDataset<FriendsEmotionItem>) -> eyre::Result<InMemDataset<EmotionItem>> {
    let embeddings = cache.embed(dataset.iter().map(|v| v.text))?;
    let test_dataset_vec = dataset.iter().zip(embeddings).map(|(item, embedding)| {
        EmotionItem {
            text_embedding: embedding,
            label: NEW_CLASSES.iter().position(|v| *v == item.label).unwrap(),
        }
    }).collect();
    let new_dataset = InMemDataset::new(test_dataset_vec);

    cache.save_cache()?;
    Ok(new_dataset)
}


fn get_emotion_map() -> HashMap<String, usize> {
    HashMap::from([
        ("admiration".to_string(), 1),
        ("amusement".to_string(), 2),
        ("anger".to_string(), 4),
        ("annoyance".to_string(), 4),
        ("approval".to_string(), 1),
        ("caring".to_string(), 1),
        ("confusion".to_string(), 1),
        ("curiosity".to_string(), 3),
        ("desire".to_string(), 1),
        ("disappointment".to_string(), 5),
        ("disapproval".to_string(), 4),
        ("disgust".to_string(), 6),
        ("embarrassment".to_string(), 6),
        ("excitement".to_string(), 2),
        ("fear".to_string(), 7),
        ("gratitude".to_string(), 2),
        ("grief".to_string(), 5),
        ("joy".to_string(), 2),
        ("love".to_string(), 2),
        ("nervousness".to_string(), 7),
        ("optimism".to_string(), 2),
        ("pride".to_string(), 2),
        ("realization".to_string(), 3),
        ("relief".to_string(), 1),
        ("remorse".to_string(), 5),
        ("sadness".to_string(), 5),
        ("surprise".to_string(), 3),
        ("neutral".to_string(), 0),
    ])
}


// let emb_shape = [embeddings[0].len()];
//
// let mut embeddings2 = embeddings
//     .into_iter()
//     .map(|emb| TensorData::new(emb, emb_shape).convert::<f32>())
//     .map(|data| Tensor::<Back, 1>::from_data(data, &device))
//     .map(|tens| tens.reshape([1, emb_shape[0]]))
//     .collect::<Vec<_>>();
// let embeddings = Tensor::cat(embeddings2, 0).to_device(&device);
// infer("./artifacts", device, embeddings);
// let logits = model.forward(embeddings2.pop().unwrap());
// println!("Logits: {logits:?}");

// small_talk_ml::training::train::<MyAuto>("./artifacts", train, test, llama, TrainingConfig::new(model_cfg, AdamConfig::new()), device);