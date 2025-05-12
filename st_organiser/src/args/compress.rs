use std::sync::Arc;
use eyre::ContextCompat;
use itertools::Itertools;
use path_abs::PathInfo;
use rayon::prelude::*;
use st_http::config::SharedConfig;
use st_system::voice_manager::{VoiceManager, VoiceReference};

#[derive(clap::Args, Debug)]
pub struct CompressCommand {
    /// The name of the game-session which we want to compress
    ///
    /// All lines which are not yet compressed will be compressed to OGG Vorbis, and backups of the old files will be made
    game_name: String,
    /// Exclude a particular voice if it matches (part of) the given string.
    #[clap(long)]
    filter_exclude: Option<String>
}

impl CompressCommand {
    #[tracing::instrument(skip_all, fields(self.sample_path))]
    pub async fn run(self, config: SharedConfig) -> eyre::Result<()> {
        let game_dir = config.dirs.game_dir(&self.game_name);
        let lines_backup = game_dir.join("lines_wav_backup");
        let (game_data, line_cache, db) = st_system::session::GameData::create_or_load_from_file(&self.game_name, &config.dirs).await?;
        let shared_data = st_system::session::GameSharedData {
            game_db: db,
            config: config.dirs.clone(),
            voice_manager: Arc::new(VoiceManager::new(config.dirs.clone())),
            game_data,
            line_cache: Arc::new(tokio::sync::Mutex::new(line_cache)),
        };

        let mut lock = shared_data.line_cache.lock().await;

        for (voice, lines) in &mut lock.voice_to_line {
            if self.filter_exclude.as_ref().map(|filter| voice.name.contains(filter)).unwrap_or_default() {
                tracing::debug!(?voice, "Skipping voice as it matched the exclude filter");
                continue
            }

            let voice_line_dir = shared_data.lines_voice_path(voice);
            let dir_name = voice_line_dir.file_name().context("No filename")?.to_string_lossy();
            let backup_dir = lines_backup.join(format!("{dir_name}"));
            std::fs::create_dir_all(&backup_dir)?;

            tracing::info!(?voice, "Compressing voice lines");

            if let Err(e) = lines.iter_mut()
                .par_bridge()
                .filter(|(_, file)| file.ends_with(".wav"))
                .try_for_each(|(_, file)| {
                    let wav_path = voice_line_dir.join(&file);
                    let backup_wav = wav_path.file_name().expect("Impossible");
                    let ogg_path = wav_path.with_extension("ogg");

                    // In case the process was interrupted
                    if ogg_path.exists() {
                        *file = ogg_path.file_name().context("impossible")?.to_string_lossy().into();
                        let _ = std::fs::rename(&wav_path, backup_dir.join(backup_wav));
                        return Ok(());
                    }
                    if !wav_path.exists() {
                        return Err(eyre::eyre!("{wav_path:?} does not exist"));
                    }

                    let mut wav_file = wavers::Wav::<f32>::from_path(&wav_path)?;
                    let audio_data = st_system::postprocessing::AudioData::new(&mut wav_file)?;

                    audio_data.write_to_ogg_vorbis(&ogg_path, 0.6)?;
                    *file = ogg_path.file_name().context("impossible")?.to_string_lossy().into();

                    std::fs::rename(&wav_path, backup_dir.join(backup_wav))?;
                    Ok::<_, eyre::Error>(())
                }) {
                drop(lock);
                shared_data.save_cache().await?;
                return Err(e);
            }
        }

        drop(lock);
        shared_data.save_cache().await?;

        Ok(())
    }
}
