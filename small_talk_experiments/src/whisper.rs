use std::{collections::HashMap, ffi::c_int, path::Path, time::Instant};

use wavers::Wav;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperError, WhisperState};
use crate::audio::trim_silence;

pub fn main() -> eyre::Result<()> {
    let mut reader: Wav<f32> = wavers::Wav::from_path("NARRATOR_4017.wav")?;
    let samples = reader.read()?;

    let trimmed_samples = trim_silence(&samples, reader.n_channels() as usize, 0.01); // Adjust threshold as needed
    println!("Trimmed length: {}", trimmed_samples.len());
    
    let mut whisper = WhisperTranscribe::new(r"G:\TTS\small-talk-data\models\whisper\ggml\ggml-base.en-q8_0.bin", 8)?;
    let prompt = "A skeleton is walking across the wastes. He moves with pep in his step, humming a tune. His skull bobs in chorus with the humming, making the coins inside his head clink rhythmically. Next to him, nightmarish horses of flame and shadow draw a cart loaded with valuable-looking items. Upon seeing you, the skeleton offers a dramatic and hearty wave.";
    let now = Instant::now();
    let data = whisper.infer(&trimmed_samples, reader.n_channels())?;
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

    pub fn infer(&mut self, samples: &[f32], n_channels: u16) -> eyre::Result<String> {
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

        let new_samples = convert_any_to_mono(samples, n_channels as usize);
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