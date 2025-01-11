use std::time::Instant;
use hound;
use wavers::Wav;

mod f5_rs;

fn main() -> eyre::Result<()> {
    let mut reader: Wav<f32> = wavers::Wav::from_path("input.wav")?;
    let samples = reader.read()?;

    let trimmed_samples = trim_silence(&samples, reader.n_channels() as usize, 0.01); // Adjust threshold as needed
    println!("Trimmed length: {}", trimmed_samples.len());
    
    wavers::write("output.wav", &trimmed_samples, reader.sample_rate(), reader.n_channels())?;
    
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