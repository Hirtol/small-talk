use std::sync::Arc;
use std::time::Duration;
use eyre::ContextCompat;
use itertools::Itertools;
use path_abs::PathInfo;
use rayon::prelude::*;
use tokio::time::sleep;
use st_application::config::SharedConfig;
use st_application::SmallTalkApplication;
use st_data::voice::VoiceReference;
use st_system::emotion::EmotionBackend;
use st_system::rvc_backends::RvcCoordinator;
use st_system::rvc_backends::seedvc::local::{LocalSeedHandle, LocalSeedVcConfig};
use st_system::tts_backends::TtsCoordinator;
use st_system::{PostProcessing, RvcModel, RvcOptions, TtsModel, TtsSystem, TtsVoice, VoiceLine};
use st_system::tts_backends::echo_tts::EchoTtsHandle;
use st_system::tts_backends::indextts::IndexTtsHandle;
use st_system::voice_manager::{VoiceManager};
use tracing::Span;
use tracing_indicatif::{span_ext::IndicatifSpanExt, style::ProgressStyle};
use crate::args::ClapTtsModel;

#[derive(clap::Args, Debug)]
pub struct ReassignCommand {
    /// The name of the game-session which contains the voice we want to change.
    pub game_name: String,
    /// The voice to change
    pub voice: String,
    /// The location, either 'global' or '{GAME_NAME}'
    pub voice_location: String,
    /// Name of the new voice
    #[clap(long)]
    pub target_voice: String,
    /// The location, either 'global' or '{GAME_NAME}'
    #[clap(long)]
    pub target_location: String,
    /// The TTS Model to use for the re-generation
    #[clap(long)]
    pub model: ClapTtsModel,
    /// Whether to use RVC
    #[clap(long)]
    pub rvc: bool,
}

impl ReassignCommand {
    #[tracing::instrument(skip_all, fields(self.sample_path))]
    pub async fn run(self, config: SharedConfig) -> eyre::Result<()> {
        let tts_sys = SmallTalkApplication::new(&config).await?.tts_system;
        let game_sess = tts_sys.get_or_start_session(&self.game_name).await?;

        let new_voice = VoiceReference {
            name: self.target_voice,
            location: self.target_location.into(),
        };
        let source_voice = VoiceReference {
            name: self.voice,
            location: self.voice_location.into(),
        };
        let assigned_voices = game_sess.character_voices().await?;
        let lines_to_redo = game_sess.voice_lines(&source_voice).await?;

        for (character, voice) in assigned_voices {
            if voice != source_voice {
                continue;
            }

            tracing::info!(?character, old_voice=?voice, ?new_voice, "Reassigned character voice");

            game_sess.force_character_voice(character, new_voice.clone()).await?;
        }

        tracing::info!(todo=lines_to_redo.len(), "Regenerating lines");

        let process_span = tracing::info_span!("process_line_request");
        process_span.pb_set_style(&ProgressStyle::with_template(
            "{wide_bar} {pos}/{len} {msg} ETA {eta_precise}",
        )?);
        process_span.pb_set_length(lines_to_redo.len() as u64);
        process_span.pb_set_message("Processing lines");
        let _guard = process_span.enter();

        let mut voice_lines = lines_to_redo.into_iter().map(|line| {
            VoiceLine {
                line,
                person: TtsVoice::ForceVoice(new_voice.clone()),
                model: self.model.into(),
                force_generate: true,
                post: Some(PostProcessing {
                    verify_percentage: None,
                    trim_silence: true,
                    normalise: true,
                    rvc: self.rvc.then_some(RvcOptions {
                        model: RvcModel::SeedVc,
                        high_quality: true,
                    }),
                }),
            }
        }).collect_vec();

        while let Some(line) = voice_lines.pop() {
            if let Err(e) = game_sess.request_tts(line.clone()).await {
                // Retry failed ones
                tracing::debug!(?e, "Pushing {line:?} onto retry queue");
                voice_lines.push(line)
            } else {
                process_span.pb_inc(1)
            }
        }

        Ok(())
    }
}
