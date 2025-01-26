//! Speech-to-text functionality

use std::path::Path;

use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperState};
pub struct WhisperTranscribe {
    _whisper: WhisperContext,
    state: WhisperState,
    cpu_concurrency: u16,
}

impl WhisperTranscribe {
    /// Create a new Whisper instance, loading the given model and using at most `cpu_threads` for the computations.
    ///
    /// If the crate was compiled with the `cuda` feature enabled this will automatically use the GPU.
    pub fn new(model_path: impl AsRef<Path>, cpu_threads: u16) -> eyre::Result<Self> {
        whisper_rs::install_whisper_tracing_trampoline();
        // load a context and model, always use GPU if we can.
        let ctx = WhisperContext::new_with_params(
            &model_path.as_ref().to_string_lossy(),
            WhisperContextParameters {
                use_gpu: true,
                ..Default::default()
            },
        )?;

        let state = ctx.create_state()?;

        Ok(Self {
            _whisper: ctx,
            cpu_concurrency: cpu_threads,
            state,
        })
    }

    /// Transcribe the given `wav_file` (expected `.wav`).
    pub fn transcribe_file(&mut self, wav_file: impl AsRef<Path>) -> eyre::Result<String> {
        let mut reader: wavers::Wav<f32> = wavers::Wav::from_path(wav_file)?;
        self.infer(&reader.read()?, reader.n_channels(), reader.sample_rate() as u32)
    }

    /// Infer the text spoken in the given audio.
    ///
    /// The samples should be given with interleaved channels.
    pub fn infer(&mut self, samples: &[f32], n_channels: u16, sampling_rate: u32) -> eyre::Result<String> {
        // 16 KHz sample rate expected, may need to re-sample.
        const WHISPER_SAMPLE_RATE: u32 = 16_000;
        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });

        // Set english as our main language, consider switching.
        params.set_language(Some(&"en"));
        params.set_n_threads(self.cpu_concurrency as i32);
        params.set_no_timestamps(true);

        // We also explicitly disable anything that prints to stdout
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);

        let mut new_samples = convert_any_to_mono(samples, n_channels as usize);

        if sampling_rate != WHISPER_SAMPLE_RATE {
            // We've converted the audio to mono already, so it's only 1 channel.
            new_samples = audio_resample(&new_samples, sampling_rate, WHISPER_SAMPLE_RATE, 1);
        }

        self.state.full(params, &new_samples[..])?;

        // We set `single_segment` to true so we can just get the first.
        let num_segments = self.state.full_n_segments()?;

        let text = (0..num_segments)
            .map(|i| self.state.full_get_segment_text(i))
            .collect::<Result<String, _>>()?;

        Ok(text)
    }
}

/// Convert the given, potentially multi-channel, audio into a mono-channel sequence.
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

fn audio_resample(
    data: &[f32],
    from_rate: u32,
    to_rate: u32,
    channels: u16,
) -> Vec<f32> {
    use samplerate::{convert, ConverterType};
    convert(
        from_rate as _,
        to_rate as _,
        channels as _,
        ConverterType::SincBestQuality,
        data,
    )
    .unwrap_or_default()
}