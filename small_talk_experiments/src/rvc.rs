use std::borrow::Cow;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;
use fundsp::hacker::*;
use fundsp::wave::Wave;
use reqwest::blocking::{multipart, Client};
use serde::Serialize;
use wavers::Wav;

pub fn main() -> eyre::Result<()> {
    let mut reader: Wav<f32> = wavers::Wav::from_path("input.wav")?;
    let target_voice = Path::new("Anger_0.wav");
    let samples = reader.read()?;

    let trimmed_samples = trim_silence(&samples, reader.n_channels() as usize, 0.01); // Adjust threshold as needed
    println!("Trimmed length: {}", trimmed_samples.len());

    // Define the metadata
    let metadata = SoundMetadata {
        sample_rate: reader.sample_rate() as u32,
        channels: reader.n_channels() as u8,
        target_voice: target_voice.canonicalize()?,
    };

    let body_data: &[u8] = bytemuck::cast_slice(&trimmed_samples);
    // Create a multipart form
    let form = multipart::Form::new()
        // Add the sound file as a part
        .part(
            "sound_samples",
            multipart::Part::bytes(body_data.to_vec())
                .file_name("sound_file.raw") // Optional: specify file name
                .mime_str("application/octet-stream")?, // Update MIME type if needed
        )
        .text("sample_rate", metadata.sample_rate.to_string())
        .text("channels", metadata.channels.to_string())
        .text("target_voice", metadata.target_voice.to_string_lossy().to_string());
        // Add the metadata as a part
        // .text("metadata", serde_json::to_string(&metadata)?);

    // Make the POST request
    let client = Client::new();
    let now = Instant::now();
    let response = client
        .post("http://127.0.0.1:9999/api/rvc") // Update with your endpoint
        .multipart(form)
        .send()?;
    println!("Took: {:?}", now.elapsed());
    // let txt = response.text()?;
    let now = Instant::now();
    // println!("TEXT: {txt}");
    let content = response.bytes()?;
    println!("Took: {:?}", now.elapsed());
    std::fs::write("rvc.wav", content)?;
    println!("Wrote RVC");

    Ok(())
}

#[derive(Serialize)]
struct SoundMetadata {
    sample_rate: u32,
    channels: u8,
    target_voice: PathBuf
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