use std::{collections::HashMap, ffi::c_int, path::Path, time::Instant};

use wavers::Wav;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperError, WhisperState};
use crate::audio::trim_silence;

pub fn main() -> eyre::Result<()> {
    let mut reader: Wav<f32> = wavers::Wav::from_path("Anger_0.wav")?;
    let samples = reader.read()?;

    let trimmed_samples = trim_silence(&samples, reader.n_channels() as usize, 0.01); // Adjust threshold as needed
    println!("Trimmed length: {}", trimmed_samples.len());
    let trimmed_samples2 = convert_any_to_mono(trimmed_samples, reader.n_channels() as usize);
    let trimmed_samples2: Vec<i16> = trimmed_samples2.iter().map(|s| (*s * i16::MAX as f32) as i16).collect();
    wavers::write("output.wav", &trimmed_samples2, reader.sample_rate(), 1)?;

    let mut whisper = WhisperTranscribe::new(r"G:\TTS\small-talk-data\models\whisper\ggml\ggml-large-v3-turbo-q5_0.bin", 8)?;
    let prompt = "A skeleton is walking across the wastes. He moves with pep in his step, humming a tune. His skull bobs in chorus with the humming, making the coins inside his head clink rhythmically. Next to him, nightmarish horses of flame and shadow draw a cart loaded with valuable-looking items. Upon seeing you, the skeleton offers a dramatic and hearty wave.";
    let now = Instant::now();
    let data = whisper.infer(&trimmed_samples, reader.n_channels(), reader.sample_rate() as u32)?;
    println!("{}", data);
    println!("Took: {:?}", now.elapsed());

    let distance = strsim::levenshtein(&data, prompt);
    let levenshtein_ratio = 1.0 - (distance as f64 / prompt.chars().count() as f64);
    println!("Levenshtein Ratio: {:?}", levenshtein_ratio);
    Ok(())
}

struct WhisperTranscribe {
    whisper: WhisperContext,
    state: WhisperState,
    cpu_concurrency: u16,
}

impl WhisperTranscribe {
    pub fn new(model_path: impl AsRef<str>, cpu_threads: u16) -> eyre::Result<Self> {
        whisper_rs::install_whisper_tracing_trampoline();
        // load a context and model, always use GPU if we can.
        let ctx = WhisperContext::new_with_params(model_path.as_ref(), WhisperContextParameters {
            use_gpu: true,
            ..Default::default()
        })?;

        let mut state = ctx.create_state()?;

        Ok(Self {
            whisper: ctx,
            cpu_concurrency: cpu_threads,
            state,
        })
    }

    pub fn infer(&mut self, samples: &[f32], n_channels: u16, sampling_rate: u32) -> eyre::Result<String> {
        // 16 KHz sample rate expected, may need to re-sample.
        const WHISPER_SAMPLE_RATE: u32 = 16_000;
        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });

        // and set the language to translate to to english
        params.set_language(Some(&"en"));
        params.set_n_threads(self.cpu_concurrency as c_int);
        params.set_no_timestamps(true);

        // we also explicitly disable anything that prints to stdout
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);

        let mut new_samples = convert_any_to_mono(samples, n_channels as usize);

        if sampling_rate != WHISPER_SAMPLE_RATE {
            new_samples = audio_resample(&new_samples, sampling_rate, WHISPER_SAMPLE_RATE, n_channels);
        }

        self.state.full(params, &new_samples[..])?;

        // We set `single_segment` to true so we can just get the first.
        let num_segments = self.state
        .full_n_segments()
        .expect("failed to get number of segments");
        let mut segment = String::new();
        for i in 0..num_segments {
            let new_segment = self.state
                .full_get_segment_text(i)
                .expect("failed to get segment");
            segment.push_str(&new_segment);
        }

        Ok(segment)
    }
}


pub fn audio_resample(
    data: &[f32],
    sample_rate0: u32,
    sample_rate: u32,
    channels: u16,
) -> Vec<f32> {
    use samplerate::{convert, ConverterType};
    convert(
        sample_rate0 as _,
        sample_rate as _,
        channels as _,
        ConverterType::SincBestQuality,
        data,
    )
    .unwrap_or_default()
}

fn convert_any_to_mono(samples: &[f32], channels: usize) -> Vec<f32> {
    if channels == 1 {
        samples.to_vec()
    } else {
        samples
        .chunks_exact(channels)
        .map(|x| x.iter().sum::<f32>() / channels as f32)
        .collect()
    }
}

// #[derive(Debug, Clone, Default)]
// pub struct LineCache {
//     /// Voice -> Line voiced -> file name
//     pub voice_to_line: HashMap<VoiceReference, HashMap<String, String>>,
// }

// // Needed in order to properly handle VoiceReference
// impl Serialize for LineCache {
//     fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
//     where
//         S: Serializer,
//     {
//         // Create a temporary HashMap<String, HashMap<String, String>>
//         let transformed: HashMap<String, HashMap<String, String>> = self
//             .voice_to_line
//             .iter()
//             .map(|(key, value)| {
//                 let key_str = match &key.location {
//                     VoiceDestination::Global => format!("global_{}", key.name),
//                     VoiceDestination::Game(game_name) => format!("game_{game_name}_{}", key.name),
//                 };
//                 (key_str, value.clone())
//             })
//             .collect();

//         transformed.serialize(serializer)
//     }
// }

// impl<'de> Deserialize<'de> for LineCache {
//     fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
//     where
//         D: Deserializer<'de>,
//     {
//         // Deserialize into a temporary HashMap<String, HashMap<String, String>>
//         let raw_map: HashMap<String, HashMap<String, String>> =
//             HashMap::deserialize(deserializer)?;

//         // Convert back to HashMap<VoiceReference, HashMap<String, String>>
//         let voice_to_line = raw_map
//             .into_iter()
//             .map(|(key, value)| {
//                 let (location, name) = if let Some(rest) = key.strip_prefix("global_") {
//                     (VoiceDestination::Global, rest.to_string())
//                 } else if let Some(rest) = key.strip_prefix("game_") {
//                     let (game_name, character) = rest.split_once("_").ok_or_else(|| D::Error::custom("No game identifier found"))?;
//                     (VoiceDestination::Game(game_name.into()), character.to_string())
//                 } else {
//                     return Err(serde::de::Error::custom(format!("Invalid key format: {}", key)));
//                 };

//                 Ok((VoiceReference { name, location }, value))
//             })
//             .collect::<Result<HashMap<_, _>, D::Error>>()?;

//         Ok(LineCache { voice_to_line })
//     }
// }