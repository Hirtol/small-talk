//! Audio post-processing for generated TTS files.


/// Remove leading/trailing silences in the given audio.
/// 
/// Assumes interleaved channel samples in order to correctly chunk the audio.
pub fn trim_silence(audio_samples: &[f32], channel_count: usize, silence_threshold: f32) -> &[f32] {
    trim_trail(trim_lead(audio_samples, channel_count, silence_threshold), channel_count, silence_threshold)
}

/// Remove leading silences in the given audio.
///
/// Assumes interleaved channel samples in order to correctly chunk the audio.
pub fn trim_lead(audio_samples: &[f32], channel_count: usize, silence_threshold: f32) -> &[f32] {
    let mut start = audio_samples.iter()
        .position(|sample| sample.abs() > silence_threshold)
        .unwrap_or(0);

    // Back up to avoid offsetting channels in case only one channel has audio.
    let remainder = start % channel_count;
    start -= remainder;
    
    &audio_samples[start..]
}

/// Remove trailing silences in the given audio.
///
/// Assumes interleaved channel samples in order to correctly chunk the audio.
pub fn trim_trail(audio_samples: &[f32], channel_count: usize, silence_threshold: f32) -> &[f32] {
    let mut end = audio_samples.iter()
        .rposition(|sample| sample.abs() > silence_threshold)
        .unwrap_or(0);

    // Back up to avoid offsetting channels in case only one channel has audio.
    let remainder = end % channel_count;
    end -= remainder;

    &audio_samples[..end]
}