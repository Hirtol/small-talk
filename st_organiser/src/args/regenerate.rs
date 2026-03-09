use crate::args::ClapTtsModel;
use itertools::Itertools;
use sea_orm::{
    prelude::Expr, sea_query::{Alias, ExprTrait, Query}, ColumnTrait, DbBackend, EntityTrait, JoinType, QueryFilter, QuerySelect,
    SelectColumns,
    Statement,
};
use st_application::{config::SharedConfig, SmallTalkApplication};
use st_data::voice::VoiceReference;
use st_system::{
    session::{db, db::SessionDb, GameSessionHandle}, PostProcessing, RvcModel, RvcOptions, TtsSystem, TtsVoice,
    VoiceLine,
};
use tracing::Span;
use tracing_indicatif::{span_ext::IndicatifSpanExt, style::ProgressStyle};

#[derive(clap::Args, Debug)]
pub struct RegenerateCommand {
    /// The name of the game-session which contains the voice lines
    game_name: String,
    #[clap(long)]
    model: ClapTtsModel,
    /// Whether to use RVC for the post-generation step.
    #[clap(long)]
    rvc: bool,
    #[clap(subcommand)]
    sub_command: RegenerateSubCommand,
}

#[derive(clap::Subcommand, Debug, Clone)]
pub enum RegenerateSubCommand {
    /// Regenerate lines for a specific voice
    Voice(RegenerateVoice),
    /// Find all missing lines (dialogue present in the dialogue table, but without associated voice_lines)
    MissingLines,
}

#[derive(clap::Args, Debug, Clone)]
pub struct RegenerateVoice {
    /// The voice to change (optional: if not provided, all matching voices will be regenerated)
    #[clap(long, short)]
    voice: Option<Vec<String>>,
    /// The location, either 'global' or '{GAME_NAME}' (optional)
    #[clap(long, short)]
    voice_location: Option<String>,
    /// Various filters for the lines to regenerate
    #[command(flatten)]
    patterns: RegenerateFilters,
}

#[derive(clap::Args, Debug, Clone)]
pub struct RegenerateFilters {
    /// SQLite LIKE pattern for dialogue text (e.g. "%there's%")
    #[clap(long)]
    dialogue_pattern: Option<String>,
    /// SQLite ID voice_line id which marks the end of eligible regenerations.
    #[clap(long)]
    max_id: Option<usize>,
    /// SQLite ID voice_line id which marks the start of eligible regenerations.
    #[clap(long)]
    min_id: Option<usize>,
    /// Voices to exclude from any other applicable patterns
    #[clap(long)]
    exclude_voice: Option<Vec<String>>,
    /// SQLite LIKE pattern for file name (e.g. "%.wav")
    #[clap(long)]
    file_pattern: Option<String>,
}

impl RegenerateCommand {
    #[tracing::instrument(skip_all, fields(self.sample_path))]
    pub async fn run(self, config: SharedConfig) -> eyre::Result<()> {
        match self.sub_command.clone() {
            RegenerateSubCommand::MissingLines => self.handle_missing_lines(config).await,
            RegenerateSubCommand::Voice(sub_command) => self.handle_voice_regen(config, sub_command).await,
        }
    }

    #[tracing::instrument(skip_all)]
    async fn handle_missing_lines(self, config: SharedConfig) -> eyre::Result<()> {
        let tts_sys = SmallTalkApplication::new(&config).await?.tts_system;
        let game_sess = tts_sys.get_or_start_session(&self.game_name).await?;

        let missing = Self::find_all_missing_voicelines(game_sess.session_db()).await?;

        tracing::info!("Missing: {:#?} lines", missing.len());

        self.process_line_requests(game_sess, missing).await
    }

    #[tracing::instrument(skip_all)]
    async fn handle_voice_regen(self, config: SharedConfig, voice: RegenerateVoice) -> eyre::Result<()> {
        // Handle pattern-based regeneration across all voices
        let tts_sys = SmallTalkApplication::new(&config).await?.tts_system;
        let game_sess = tts_sys.get_or_start_session(&self.game_name).await?;

        // Get all voice lines matching patterns
        let lines = Self::find_matching_lines(
            game_sess.session_db(),
            voice.voice,
            voice.voice_location,
            voice.patterns,
        )
        .await?;

        tracing::info!(todo = lines.len(), "Regenerating lines across all matching voices");

        self.process_line_requests(game_sess, lines).await
    }

    async fn process_line_requests(
        &self,
        game_session: GameSessionHandle,
        lines: Vec<(String, VoiceReference)>,
    ) -> eyre::Result<()> {
        let process_span = tracing::info_span!("process_line_request");
        process_span.pb_set_style(&ProgressStyle::with_template(
            "{wide_bar} {pos}/{len} {msg} ETA {eta_precise}",
        )?);
        process_span.pb_set_length(lines.len() as u64);
        process_span.pb_set_message("Processing lines");
        let _guard = process_span.enter();

        let mut voice_lines = lines
            .into_iter()
            .map(|(text, voice_ref)| VoiceLine {
                line: text,
                person: TtsVoice::ForceVoice(voice_ref),
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
            })
            .collect_vec();

        while let Some(line) = voice_lines.pop() {
            if let Err(e) = game_session.request_tts(line.clone()).await {
                // Retry failed ones
                tracing::debug!(?e, "Pushing {line:?} onto retry queue");
                voice_lines.push(line)
            } else {
                process_span.pb_inc(1)
            }
        }

        Ok(())
    }

    async fn find_matching_lines(
        db: SessionDb,
        voices: Option<Vec<String>>,
        voice_location: Option<String>,
        filters: RegenerateFilters,
    ) -> eyre::Result<Vec<(String, VoiceReference)>> {
        let mut condition = sea_orm::Condition::all();

        if let Some(voices) = voices {
            condition = condition.add(db::voice_lines::Column::VoiceName.is_in(voices));
        }

        if let Some(location) = voice_location {
            condition = condition.add(db::voice_lines::Column::VoiceLocation.eq(&location));
        }

        if let Some(exclude_voices) = &filters.exclude_voice {
            condition = condition.add(db::voice_lines::Column::VoiceName.is_not_in(exclude_voices))
        }

        if let Some(pattern) = &filters.dialogue_pattern {
            condition = condition.add(db::voice_lines::Column::DialogueText.like(pattern));
        }

        if let Some(pattern) = &filters.file_pattern {
            condition = condition.add(db::voice_lines::Column::FileName.like(pattern));
        }

        if let Some(cutoff) = &filters.max_id {
            condition = condition.add(db::voice_lines::Column::Id.lt(*cutoff as u64))
        }

        if let Some(cutoff) = &filters.min_id {
            condition = condition.add(db::voice_lines::Column::Id.gte(*cutoff as u64))
        }

        let results: Vec<(String, String, String)> = db::voice_lines::Entity::find()
            .select_only()
            .columns([
                db::voice_lines::Column::DialogueText,
                db::voice_lines::Column::VoiceName,
                db::voice_lines::Column::VoiceLocation,
            ])
            .filter(condition)
            .into_tuple()
            .all(db.reader())
            .await?;

        Ok(results
            .into_iter()
            .map(|(text, name, location)| {
                (
                    text,
                    VoiceReference {
                        name,
                        location: location.into(),
                    },
                )
            })
            .collect())
    }

    async fn find_all_missing_voicelines(db: SessionDb) -> eyre::Result<Vec<(String, VoiceReference)>> {
        use sqlx::Row;
        // Could not figure out how to express the below in Sea-orm/seaquery, so raw sqlx it is
        let rows = sqlx::query(
            r#"
        SELECT
        DISTINCT d.dialogue_text, c.voice_name, c.voice_location
        FROM dialogue d
                 LEFT JOIN characters c
                           ON c.id = d.character_id
                 LEFT JOIN voice_lines v
                           ON d.dialogue_text = v.dialogue_text
                                  AND v.voice_name = c.voice_name
                                  AND v.voice_location = c.voice_location
        WHERE v.id IS NULL
        "#,
        )
        .fetch_all(db.reader().get_sqlite_connection_pool())
        .await?;

        Ok(rows
            .into_iter()
            .map(|row| {
                (
                    row.get(0),
                    VoiceReference {
                        name: row.get(1),
                        location: row.get::<String, _>(2).into(),
                    },
                )
            })
            .collect())
    }
}
