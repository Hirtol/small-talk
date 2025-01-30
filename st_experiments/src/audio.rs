use std::fs::File;
use std::num::{NonZeroU32, NonZeroU8};
use fundsp::hacker::*;
use fundsp::wave::Wave;
use itertools::Itertools;
use vorbis_rs::{VorbisBitrateManagementStrategy, VorbisEncoderBuilder};
use wavers::Wav;

const VORBIS_BLOCK_LEN: usize = 4096;

pub fn main() -> eyre::Result<()> {
    let mut reader: Wav<f32> = wavers::Wav::from_path("input.wav")?;
    let samples = reader.read()?;

    let trimmed_samples = trim_silence(&samples, reader.n_channels() as usize, 0.01); // Adjust threshold as needed
    println!("Trimmed length: {}", trimmed_samples.len());
    
    let buf = fundsp::hacker::BufferVec::new(reader.n_channels() as usize);
    let mut trimmed_samples_i16: Vec<i16> = trimmed_samples.iter().map(|s| (*s * i16::MAX as f32) as i16).collect();
    wavers::write("output.wav", &trimmed_samples_i16, reader.sample_rate(), reader.n_channels())?;

    let mut transcoded_ogg = Vec::new();
    let mut encoder = VorbisEncoderBuilder::new(
        NonZeroU32::new(reader.sample_rate() as u32).unwrap(),
        NonZeroU8::new(reader.n_channels() as u8).unwrap(),
        &mut transcoded_ogg
    )?;
    encoder.bitrate_management_strategy(VorbisBitrateManagementStrategy::QualityVbr {
        target_quality: 0.8
    });
    let mut encoder = encoder.build()?;
    let mut output_buffers = vec![Vec::new(); reader.n_channels() as usize];
    for chunk in &trimmed_samples.into_iter().chunks(reader.n_channels() as usize) {
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

    std::fs::write("output.ogg", transcoded_ogg)?;


    let wav = fundsp::hacker::wave::Wave::load("output.wav")?;
    let high_pass = highpass_hz(80.0, 1.0);
    let low_pass = lowpass_hz(12000.0, 1.0);

    let de_esser = (bandpass_hz(6000.0, 1.0));
    let mut processing_graph = high_pass >> low_pass;


    let out = wav.filter(wav.duration(), &mut processing_graph);

    out.write_wav16(&mut File::create("out_filter.wav")?)?;
    
    Ok(())
}

/// Remove leading/trailing silences in the given audio.
///
/// Assumes interleaved channel samples in order to correctly chunk the audio.
pub fn trim_silence(audio_samples: &[f32], channel_count: usize, silence_threshold: f32) -> &[f32] {
    trim_trail(trim_lead(audio_samples, channel_count, silence_threshold), channel_count, silence_threshold)
}

pub fn trim_lead(audio_samples: &[f32], channel_count: usize, silence_threshold: f32) -> &[f32] {
    let mut start = audio_samples.iter()
        .position(|sample| sample.abs() > silence_threshold)
        .unwrap_or(0);

    // Back up to avoid offsetting channels in case only one channel has audio.
    let remainder = start % channel_count;
    start -= remainder;

    &audio_samples[start..]
}

pub fn trim_trail(audio_samples: &[f32], channel_count: usize, silence_threshold: f32) -> &[f32] {
    let mut end = audio_samples.iter()
        .rposition(|sample| sample.abs() > silence_threshold)
        .unwrap_or(0);

    // Back up to avoid offsetting channels in case only one channel has audio.
    let remainder = end % channel_count;
    end -= remainder;

    &audio_samples[..end]
}