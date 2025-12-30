use crate::AudioData;
use std::time::Duration;

/// Stitch with fixed silent gaps (gap_ms) between chunks (no crossfade).
pub fn stitch_with_gaps(
    chunks: impl ExactSizeIterator<Item = AudioData>,
    gap: Duration,
) -> eyre::Result<AudioData> {
    if chunks.len() <= 1 {
        return chunks
            .into_iter()
            .next()
            .ok_or_else(|| eyre::eyre!("Can't stitch 0 samples"));
    }

    let mut silence = Vec::new();
    let mut target_rate = 0;
    let mut target_channels = 0;

    let mut out = Vec::new();
    for (i, p) in chunks.into_iter().enumerate() {
        if i > 0 {
            if p.sample_rate != target_rate || p.n_channels != target_channels {
                Err(eyre::eyre!(
                    "Either the sample rate or target channels don't match between two chunks"
                ))?
            }
            out.extend_from_slice(&silence);
        } else {
            target_channels = p.n_channels;
            target_rate = p.sample_rate;
            let gap_frames = (gap.as_secs_f64() * (target_rate as f64)).round() as usize;
            silence = vec![0.0f32; gap_frames * target_channels as usize];
        }
        out.extend_from_slice(&p.samples);
    }

    Ok(AudioData {
        samples: out,
        n_channels: target_channels,
        sample_rate: target_rate,
    })
}

/// Convert channel count to target (simple upmix/downmix).
/// - upmix: duplicate mono to all channels
/// - downmix: average channels to mono
pub fn convert_channels(a: &AudioData, target_channels: u16) -> AudioData {
    if a.n_channels == target_channels {
        return a.clone();
    }
    let src_ch = a.n_channels as usize;
    let dst_ch = target_channels as usize;
    let frames = a.frames_len();

    let mut out = AudioData {
        samples: vec![0.0; frames * dst_ch],
        n_channels: target_channels,
        sample_rate: a.sample_rate,
    };

    for f in 0..frames {
        let base_src = f * src_ch;
        let base_dst = f * dst_ch;

        if src_ch == 1 && dst_ch >= 1 {
            // upmix mono -> multi: duplicate
            let s = a.samples[base_src];
            for c in 0..dst_ch {
                out.samples[base_dst + c] = s;
            }
        } else if dst_ch == 1 {
            // downmix multi -> mono: simple average
            let mut sum = 0.0f32;
            for c in 0..src_ch {
                sum += a.samples[base_src + c];
            }
            out.samples[base_dst] = sum / (src_ch as f32);
        } else {
            // general case: if channel counts differ but neither is 1,
            // copy as many channels as possible and zero others.
            for c in 0..dst_ch {
                let src_c = if c < src_ch { c } else { src_ch - 1 };
                out.samples[base_dst + c] = a.samples[base_src + src_c];
            }
        }
    }

    out
}

/// Linear resample per-channel, interleaved samples.
/// Produces frames = round(src_frames * target_rate / src_rate)
pub fn resample_audio(a: &AudioData, target_rate: u32) -> AudioData {
    if a.sample_rate == target_rate {
        return a.clone();
    }
    if a.sample_rate == 0 || target_rate == 0 || a.n_channels == 0 {
        return AudioData {
            samples: vec![],
            n_channels: a.n_channels,
            sample_rate: target_rate,
        };
    }

    let src_rate = a.sample_rate as f64;
    let dst_rate = target_rate as f64;
    let src_frames = a.frames_len();
    if src_frames == 0 {
        return AudioData {
            samples: vec![],
            n_channels: a.n_channels,
            sample_rate: target_rate,
        };
    }

    let dst_frames_f = (src_frames as f64) * (dst_rate / src_rate);
    let dst_frames = dst_frames_f.max(1.0).round() as usize;
    let ch = a.n_channels as usize;

    let mut out = AudioData {
        samples: vec![0.0f32; dst_frames * ch],
        n_channels: a.n_channels,
        sample_rate: target_rate,
    };

    for dst_f in 0..dst_frames {
        // Map dst frame to source position in frames (floating)
        let src_pos = (dst_f as f64) * (src_frames as f64) / (dst_frames as f64);
        let idx0 = src_pos.floor() as isize;
        let frac = src_pos - (idx0 as f64);

        let i0 = idx0.max(0) as usize;
        let i1 = if i0 + 1 < src_frames { i0 + 1 } else { i0 };

        let base_out = dst_f * ch;
        let base_i0 = i0 * ch;
        let base_i1 = i1 * ch;

        for c in 0..ch {
            let s0 = a.samples[base_i0 + c];
            let s1 = a.samples[base_i1 + c];
            let val = (1.0 - frac as f32) * s0 + (frac as f32) * s1;
            out.samples[base_out + c] = val;
        }
    }

    out
}