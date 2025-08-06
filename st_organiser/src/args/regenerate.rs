use crate::args::ClapTtsModel;
use st_http::config::SharedConfig;
use st_system::{VoiceLine, TtsVoice, PostProcessing, RvcOptions, RvcModel, TtsSystem};
use itertools::Itertools;

#[derive(clap::Args, Debug)]
pub struct RegenerateCommand {
    /// The name of the game-session which contains the voice lines
    game_name: String,
    /// The voice to change (optional: if not provided, all matching voices will be regenerated)
    #[clap(long, short)]
    voice: Option<String>,
    /// The location, either 'global' or '{GAME_NAME}' (optional)
    #[clap(long, short)]
    voice_location: Option<String>,
    #[clap(long)]
    model: ClapTtsModel,
    /// SQLite LIKE pattern for dialogue text (e.g. "%there's%")
    #[clap(long)]
    dialogue_pattern: Option<String>,
    /// SQLite LIKE pattern for file name (e.g. "%.wav")
    #[clap(long)]
    file_pattern: Option<String>,
}

impl RegenerateCommand {
    #[tracing::instrument(skip_all, fields(self.sample_path))]
    pub async fn run(self, config: SharedConfig) -> eyre::Result<()> {
        if let (Some(voice), Some(voice_location)) = (self.voice, self.voice_location) {
            // Use ReassignCommand for voice-specific regeneration
            let command = super::reassign::ReassignCommand {
                game_name: self.game_name,
                voice: voice.clone(),
                voice_location: voice_location.clone(),
                target_voice: voice,
                target_location: voice_location,
                model: self.model,
            };
            command.run(config).await
        } else {
            // Handle pattern-based regeneration across all voices
            let tts_sys = super::reassign::create_tts_system(config)?;
            let game_sess = tts_sys.get_or_start_session(&self.game_name).await?;
            
            // Get all voice lines matching patterns
            let lines = game_sess.voice_lines_by_filters(
                self.dialogue_pattern.as_deref(),
                self.file_pattern.as_deref()
            ).await?;

            tracing::info!(todo=lines.len(), "Regenerating lines across all matching voices");

            let mut voice_lines = lines.into_iter().map(|(text, voice_ref)| {
                VoiceLine {
                    line: text,
                    person: TtsVoice::ForceVoice(voice_ref),
                    model: self.model.into(),
                    force_generate: true,
                    post: Some(PostProcessing {
                        verify_percentage: None,
                        trim_silence: true,
                        normalise: true,
                        rvc: Some(RvcOptions {
                            model: RvcModel::SeedVc,
                            high_quality: true,
                        }),
                    }),
                }
            }).collect_vec();

            while let Some(line) = voice_lines.pop() {
                if let Err(_) = game_sess.request_tts(line.clone()).await {
                    // Retry failed ones
                    tracing::debug!("Pushing {line:?} onto retry queue");
                    voice_lines.push(line)
                }
            }

            Ok(())
        }
    }
}
