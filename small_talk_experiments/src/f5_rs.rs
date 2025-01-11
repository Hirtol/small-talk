//! In-progress port of F5-tts to Rust using the ONNX implementation
use burn::{
    data::dataset::{
        Dataset,
    },
    optim::AdamConfig,
    prelude::{Backend, Config, Module},
    record::{CompactRecorder, Recorder},
};
use itertools::Itertools;
use std::{
    collections::HashMap
    ,
    time::Instant,
};
use fon::samp::Samp16;
use hound::{SampleFormat, WavSpec, WavWriter};
use ort::execution_providers::{CUDAExecutionProvider, DirectMLExecutionProvider};
use crate::f5_rs;

fn main_test() -> eyre::Result<()> {
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

    let gen_text = f5_rs::tokenize_text(vec![&gen_text]);
    //let text_ids = list_str_to_idx(gen_text,)

    let pre_process_inputs = ort::inputs! {
        "audio" => ([1usize], final_audio),
        // "text_ids" => vec![],
        "max_duration" => ([1usize], vec![max_duration]),
    }?;

    Ok(())
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