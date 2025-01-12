//! Audio post-processing for generated TTS files.

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
pub fn loudness_normalise(audio_samples: &mut [f32], sample_rate: i32, channel_count: u16) {
    let mut ebur128 = ebur128::EbuR128::new(channel_count as u32, sample_rate as u32, ebur128::Mode::I)
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
