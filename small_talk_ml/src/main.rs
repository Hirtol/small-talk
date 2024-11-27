use burn::{
    backend::{wgpu::WgpuDevice, Autodiff, Wgpu},
    data::dataset::{
        transform::ComposedDataset,
        Dataset, HuggingfaceDatasetLoader, InMemDataset, SqliteDataset,
    },
    optim::AdamConfig,
    prelude::{Backend, Config, Module},
    record::{CompactRecorder, Recorder},
    tensor::{activation::softmax, Tensor},
};
use itertools::Itertools;
use llama_cpp_2::{context::params::LlamaContextParams, model::params::LlamaModelParams};
use small_talk_ml::{
    embeddings::LLamaEmbedder,
    emotion_classifier::{
        data::EmotionItem,
        model::EmotionModelConfig,
        training,
        training::{FriendsEmotionItem, GoEmotionItem, LLamaTrainEmbedder, TrainingConfig},
    },
};
use std::{
    collections::HashMap
    ,
    time::Instant,
};
use std::path::{Path, PathBuf};
use burn::backend::NdArray;
use burn::backend::ndarray::NdArrayDevice;
use burn_import::onnx::ModelGen;
use fon::samp::Samp16;
use hound::{SampleFormat, WavSpec, WavWriter};
use ort::execution_providers::{CUDAExecutionProvider, DirectMLExecutionProvider};
use small_talk_ml::emotion_classifier::{model, BasicEmotionClassifier};

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
    "fear",
];

// type Back = Wgpu<f32, i32>;
type Back = NdArray<f32, i32>;
type MyAuto = Autodiff<Back>;

fn main() -> eyre::Result<()> {
    // Find our custom ONNX Runtime dylib path somehow
    // (i.e. resolving it from the root of our program's install folder)
    // let dylib_path = crate::internal::find_onnxruntime_dylib()?;
    let dylib_path = "./onnxruntime.dll";
    let ref_path = r"G:\TTS\F5-TTS-ONNX\Export_ONNX\F5_TTS\basic_ref.wav";
    let output_path = r"G:\TTS\F5-TTS-ONNX\Export_ONNX\F5_TTS\output_rust.wav";
    let ref_text = "But if you ever get tired of your decades long devotion to one another, I'm always here for a sordid affair!";
    let gen_text = "Oh honey, what are you doing? That is a stupid idea!";
    const F5_SAMPLE_RATE: u32 = 24_000;
    const HOP_LENGTH: u32 = 256;
    const NFE_STEP: u32 = 32;
    const SPEED: f64 = 1.0;
    const SEED: u32 = 9527;

    // The path should point to the `libonnxruntime` binary, which looks like:
    // - on Unix: /etc/.../libonnxruntime.so
    // - on Windows: C:\Program Files\...\onnxruntime.dll

    // Initialize ort with the path to the dylib. This **must** be called before any usage of `ort`!
    // `init_from` returns an `EnvironmentBuilder` which you can use to further configure the environment
    // before `.commit()`ing; see the Environment docs for more information on what you can configure.
    ort::init_from(dylib_path)
        .with_name("F5TTS")
        .with_telemetry(false)
        .with_execution_providers([DirectMLExecutionProvider::default().build()])
        .commit()?;
    let preprocess = r"G:\TTS\F5-TTS-ONNX\Export_ONNX\F5_TTS\F5_ONNX\F5_Preprocess.onnx";
    let session = ort::session::Session::builder()?
        .commit_from_file(preprocess)?;

    let mut audio = hound::WavReader::open(ref_path)?;
    let sample_rate = audio.spec().sample_rate;
    println!("Reading ref file with SR: {:?}", audio.spec());

    let final_audio: Vec<f32> = if sample_rate != F5_SAMPLE_RATE {
        let f32_samples: Vec<i16> = audio.samples().flatten().collect();

        let base_audio = fon::Audio::<_, 1>::with_i16_buffer(sample_rate, f32_samples);
        let mut new_audio = fon::Audio::<Samp16, 1>::with_audio(F5_SAMPLE_RATE, &base_audio);
        // 
        // let mut writer = WavWriter::create("out_resample.wav", WavSpec {
        //     channels: audio.spec().channels,
        //     bits_per_sample: 16,
        //     sample_rate: F5_SAMPLE_RATE,
        //     sample_format: SampleFormat::Int,
        // })?;
        // 
        // for sample in new_audio.as_i16_slice() {
        //     writer.write_sample(*sample)?;
        // }
        new_audio.as_i16_slice().iter().map(|i| *i as f32).collect()
    } else {
        audio.samples().flatten().map(|i: i16| i as f32).collect()
    };

    let ref_text_len = ref_text.chars().count();
    let gen_text_len = gen_text.chars().count();
    let ref_audio_len = final_audio.len() / HOP_LENGTH as usize + 1;

    let max_duration = (ref_audio_len as f64 +
        (ref_audio_len as f64 / ref_text_len as f64 * gen_text_len as f64 / SPEED))
        .ceil() as i64;
    
    let gen_text = tokenize_text(vec![&gen_text]);
    //let text_ids = list_str_to_idx(gen_text,)
    
    let pre_process_inputs = ort::inputs! {
        "audio" => ([1usize], final_audio),
        // "text_ids" => vec![],
        "max_duration" => ([1usize], vec![max_duration]),
    }?;

    // let device = NdArrayDevice::default();
    // let mut classifier: BasicEmotionClassifier<Back> = BasicEmotionClassifier::new("models/text_emotion_classifier/classifier_head", "models/text_emotion_classifier/ggml-model-Q4_k.gguf", device).unwrap();
    // for i in 0..100 {
    //     let now = std::time::Instant::now();
    //     let out = classifier.infer([
    //         "\"Johny, why are you going over there? Johny?!\"",
    //         "Oh my clowns, great... I'm going to fucking kill you",
    //         "You make me complete, happy, and whole!",
    //         "And now.. he's dead. And I'm left with nothing.",
    //         "What kind of worm is that? Eugh, no thank you.",
    //         "You'll wish you were dead when I'm done with you!"
    //     ]).unwrap();
    //
    //     println!("Data: {out:?}, took: {:?}", now.elapsed());
    // }

    Ok(())
    // infer_setup()
    // train_setup()
}

fn tokenize_text(text_list: Vec<&str>) -> Vec<Vec<String>> {
    let mut final_text_list = Vec::new();

    let merged_translations = [
        ("“", "\""), ("”", "\""), ("‘", "'"), ("’", "'"), (";", ",")
    ];

    for text in text_list {
        let mut char_list: Vec<String> = Vec::new();
        let mut translated_text = text.to_string();

        // Perform translations
        for (old, new) in &merged_translations {
            translated_text = translated_text.replace(*old, *new);
        }

        // Tokenize text
        for segment in translated_text.split_ascii_whitespace() {
            if segment.is_ascii() {
                if !char_list.is_empty() && segment.len() > 1 && !char_list.last().unwrap().ends_with([' ', ':', '\'', '"']) {
                    char_list.push(" ".to_string());
                }
                char_list.push(segment.to_string());
            } else {
                for c in segment.chars() {
                    if c.is_ascii() {
                        char_list.push(c.to_string());
                    } else {
                        char_list.push(" ".to_string());
                        // Placeholder for pinyin logic (skip for now)
                        char_list.push(c.to_string());
                    }
                }
            }
        }
        final_text_list.push(char_list);
    }

    final_text_list
}

fn list_str_to_idx(
    text: Vec<Vec<String>>,
    vocab_char_map: &HashMap<String, i32>,
    padding_value: i32
) -> Vec<Vec<i32>> {
    let mut result = Vec::new();

    for t in text {
        let mut indices: Vec<i32> = t.iter()
            .map(|c| *vocab_char_map.get(c).unwrap_or(&0))
            .collect();

        // Pad with the specified padding value
        let max_length = result.iter().map(|v: &Vec<i32>| v.len()).max().unwrap_or(0);
        while indices.len() < max_length {
            indices.push(padding_value);
        }

        result.push(indices);
    }

    result
}

pub fn infer_setup() -> eyre::Result<()> {
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

    let now = Instant::now();

    let embeddings = llama.embed(
        [
            "\"Johny, why are you going over there? Johny?!\"",
            "Oh my clowns, great... I'm going to fucking kill you",
            "You make me complete, happy, and whole!",
            "And now.. he's dead. And I'm left with nothing.",
            "What kind of worm is that? Eugh, no thank you.",
        ],
        false,
        true,
    )?;
    println!("Embedding took: {:?}", now.elapsed());

    let now = Instant::now();

    infer::<Back>(
        "./artifacts_friends_2",
        &device,
        model::embed_to_tensor(embeddings, &device),
    );

    println!("Infer took: {:?}", now.elapsed());

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

    let mut llama_cache = LLamaTrainEmbedder::new(
        "G:\\ML Models\\Embedding Models\\bge-small-en-v1.5\\ggml-model-Q4_k.gguf",
        ctx_params,
        "./training/embedding_cache.json",
    )?;

    let train: SqliteDataset<GoEmotionItem> =
        HuggingfaceDatasetLoader::new("google-research-datasets/go_emotions").dataset("train")?;
    let test: SqliteDataset<GoEmotionItem> =
        HuggingfaceDatasetLoader::new("google-research-datasets/go_emotions").dataset("test")?;

    let train_go_dataset = transform_go_item_dataset(&mut llama_cache, train)?;
    let test_go_dataset = transform_go_item_dataset(&mut llama_cache, test)?;

    let train_friends: SqliteDataset<FriendsEmotionItem> =
        HuggingfaceDatasetLoader::new("michellejieli/friends_dataset").dataset("train")?;
    let train_friend_dataset = transform_friend_dataset(&mut llama_cache, train_friends)?;

    let merged_train_dataset = ComposedDataset::new(vec![train_go_dataset, train_friend_dataset]);

    let model_cfg = EmotionModelConfig::new(384, 8);

    training::train::<MyAuto>(
        "./artifacts_friends_3",
        merged_train_dataset,
        test_go_dataset,
        TrainingConfig::new(model_cfg, AdamConfig::new()),
        device,
    );

    Ok(())
}

pub fn infer<B: Backend>(artifact_dir: &str, device: &B::Device, embedding: Tensor<B, 2>) {
    let config =
        TrainingConfig::load(format!("{artifact_dir}/config.json")).expect("Config should exist for the model");
    let record = CompactRecorder::new()
        .load(format!("{artifact_dir}/model").into(), device)
        .expect("Trained model should exist");

    let model = config.model.init::<B>(device).load_record(record);

    let now = Instant::now();
    let output = model.forward(embedding);

    let probabilities = softmax(output.clone(), 1);
    for slice in probabilities.iter_dim(0) {
        let tensor_data = slice.into_data();
        let data = tensor_data.as_slice::<f32>().unwrap();
        let top_5 = data
            .iter()
            .enumerate()
            .map(|(i, prob)| (NEW_CLASSES[i], prob))
            .sorted_by(|a, b| b.1.partial_cmp(a.1).unwrap())
            .take(5)
            .collect_vec();

        println!("Slice: {top_5:#?}");
    }
    println!("Infer took real: {:?}", now.elapsed());

}

#[tracing::instrument(skip_all)]
fn transform_go_item_dataset(
    cache: &mut LLamaTrainEmbedder,
    dataset: SqliteDataset<GoEmotionItem>,
) -> eyre::Result<InMemDataset<EmotionItem>> {
    let emotion_map = get_emotion_map();
    let embeddings = cache.embed(dataset.iter().map(|v| v.text))?;
    let test_dataset_vec = dataset
        .iter()
        .zip(embeddings)
        .map(|(item, embedding)| EmotionItem {
            text_embedding: embedding,
            label: *emotion_map.get(CLASSES[item.labels[0] as usize]).unwrap(),
        })
        .collect();
    let new_dataset = InMemDataset::new(test_dataset_vec);

    cache.save_cache()?;
    Ok(new_dataset)
}

#[tracing::instrument(skip_all)]
fn transform_friend_dataset(
    cache: &mut LLamaTrainEmbedder,
    dataset: SqliteDataset<FriendsEmotionItem>,
) -> eyre::Result<InMemDataset<EmotionItem>> {
    let embeddings = cache.embed(dataset.iter().map(|v| v.text))?;
    let test_dataset_vec = dataset
        .iter()
        .zip(embeddings)
        .map(|(item, embedding)| EmotionItem {
            text_embedding: embedding,
            label: NEW_CLASSES.iter().position(|v| *v == item.label).unwrap(),
        })
        .collect();
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
