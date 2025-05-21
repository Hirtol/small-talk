use std::fmt::{Debug, Formatter};
use wavers::Wav;
use std::path::Path;

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
        use std::io::BufWriter;
        use std::num::{NonZeroU32, NonZeroU8};
        use eyre::ContextCompat;
        use itertools::Itertools;
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

    /// Applies a single-order lowpass filter
    ///
    /// # Arguments
    /// * `cutoff_frequency` - The cutoff frequency of the lowpass filter in Hz.
    pub fn lowpass_filter(&mut self, cutoff_frequency: f32) {
        use biquad::{Biquad, Coefficients, DirectForm2Transposed, ToHertz, Type};
        let q_value = biquad::coefficients::Q_BUTTERWORTH_F32;
        let coeffs = Coefficients::<f32>::from_params(
            Type::SinglePoleLowPass,
            self.sample_rate.hz(),
            cutoff_frequency.hz(),
            q_value,
        ).expect("Failed to construct filter");

        let mut filter = DirectForm2Transposed::<f32>::new(coeffs);

        self.samples.iter_mut()
            .for_each(|x| *x = filter.run(*x));
    }
}