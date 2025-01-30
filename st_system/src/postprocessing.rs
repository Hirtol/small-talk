//! Audio post-processing for generated TTS files.

use std::fmt::{Debug, Formatter};
use std::io::BufWriter;
use std::num::{NonZeroU32, NonZeroU8};
use std::path::Path;
use eyre::ContextCompat;
use itertools::Itertools;
use wavers::Wav;

/// Remove leading/trailing silences in the given audio.
///
/// Assumes interleaved channel samples in order to correctly chunk the audio.
pub fn trim_silence(audio_samples: &mut [f32], channel_count: u16, silence_threshold: f32) -> &mut [f32] {
    trim_trail(
        trim_lead(audio_samples, channel_count, silence_threshold),
        channel_count,
        silence_threshold,
    )
}

/// Remove leading silences in the given audio.
///
/// Assumes interleaved channel samples in order to correctly chunk the audio.
pub fn trim_lead(audio_samples: &mut [f32], channel_count: u16, silence_threshold: f32) -> &mut [f32] {
    let mut start = audio_samples
        .iter()
        .position(|sample| sample.abs() > silence_threshold)
        .unwrap_or(0);

    // Back up to avoid offsetting channels in case only one channel has audio.
    let remainder = start % channel_count as usize;
    start -= remainder;

    &mut audio_samples[start..]
}

/// Remove trailing silences in the given audio.
///
/// Assumes interleaved channel samples in order to correctly chunk the audio.
pub fn trim_trail(audio_samples: &mut [f32], channel_count: u16, silence_threshold: f32) -> &mut [f32] {
    let mut end = audio_samples
        .iter()
        .rposition(|sample| sample.abs() > silence_threshold)
        .unwrap_or(0);

    // Back up to avoid offsetting channels in case only one channel has audio.
    let remainder = end % channel_count as usize;
    end -= remainder;

    &mut audio_samples[..end]
}

/// Attempt to normalise the given samples.
/// 
/// Copied from `https://github.com/sdroege/ebur128/blob/main/examples/normalize.rs`
pub fn loudness_normalise(audio_samples: &mut [f32], sample_rate: u32, channel_count: u16) {
    let mut ebur128 = ebur128::EbuR128::new(channel_count as u32, sample_rate, ebur128::Mode::I)
        .expect("Failed to create ebur128");
    let chunk_size = sample_rate; // 1s
    let target_loudness = -23.0; // EBU R128 standard target loudness

    // Compute loudness
    for chunk in audio_samples.chunks(chunk_size as usize * channel_count as usize) {
        ebur128.add_frames_f32(chunk).expect("Failed to add frames");
    }

    let global_loudness = ebur128.loudness_global().expect("Failed to get global loudness");

    // Convert dB difference to linear gain
    let gain = 10f32.powf(((target_loudness - global_loudness) / 20.0) as f32);

    for sample in audio_samples {
        let normalized_sample = (*sample * gain).clamp(-1.0, 1.0);
        *sample = normalized_sample;
    }
}

#[derive(Clone)]
pub struct AudioData {
    pub samples: Vec<f32>,
    pub n_channels: u16,
    pub sample_rate: u32,
}

impl Debug for AudioData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AudioData")
            .field("n_channels", &self.n_channels)
            .field("sample_rate", &self.sample_rate)
            .finish_non_exhaustive()
    }
}

impl AudioData {
    pub fn new(wav: &mut Wav<f32>) -> eyre::Result<Self> {
        Ok(Self {
            samples: wav.read()?.as_ref().to_vec(),
            n_channels: wav.n_channels(),
            sample_rate: wav.sample_rate() as u32,
        })
    }

    pub fn write_to_wav_file(&self, destination: &Path) -> eyre::Result<()> {
        Ok(wavers::write(destination, &self.samples, self.sample_rate as i32, self.n_channels)?)
    }

    /// Write the current [AudioData] to an OGG Vorbis file at the given path.
    ///
    /// # Arguments
    /// - `destination` - Path for the OGG Vorbis file, should have an `.ogg` extension.
    /// - `quality` - Float in the range `[-0.2, 1.0]`, `0.6` recommended
    pub fn write_to_ogg_vorbis(&self, destination: &Path, quality: f32) -> eyre::Result<()> {
        use vorbis_rs::*;
        const VORBIS_BLOCK_LEN: usize = 4096;
        let mut write_target = BufWriter::new(std::fs::File::create(destination)?);

        let mut encoder = VorbisEncoderBuilder::new(
            NonZeroU32::new(self.sample_rate).context("Need non-zero sample rate")?,
            NonZeroU8::new(self.n_channels as u8).context("Need non-zero channels")?,
            &mut write_target
        )?;
        encoder.bitrate_management_strategy(VorbisBitrateManagementStrategy::QualityVbr {
            target_quality: quality
        });
        let mut encoder = encoder.build()?;

        let mut output_buffers = vec![Vec::new(); self.n_channels as usize];
        for chunk in &self.samples.iter().chunks(self.n_channels as usize) {
            let mut should_encode = false;
            for (sample, target) in chunk.zip(output_buffers.iter_mut()) {
                target.push(*sample);
                should_encode = target.len() >= VORBIS_BLOCK_LEN;
            }

            if should_encode {
                encoder.encode_audio_block(&output_buffers)?;
                output_buffers.iter_mut().for_each(|v| v.clear());
            }
        }
        // Encode the last few samples
        encoder.encode_audio_block(&output_buffers)?;
        encoder.finish()?;

        Ok(())
    }
}
