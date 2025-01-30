use eyre::ContextCompat;
use itertools::Itertools;
use path_abs::PathInfo;
use rayon::prelude::*;
use st_http::config::SharedConfig;

#[derive(clap::Args, Debug)]
pub struct CompressCommand {
    /// The name of the game-session which we want to compress
    ///
    /// All lines which are not yet compressed will be compressed to OGG Vorbis, and backups of the old files will be made
    game_name: String,
}

impl CompressCommand {
    #[tracing::instrument(skip_all, fields(self.sample_path))]
    pub async fn run(self, config: SharedConfig) -> eyre::Result<()> {
        let lines_cache = config.dirs.game_lines_cache(&self.game_name);

        for entry in std::fs::read_dir(&lines_cache)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_dir() {
                let dir_name = path.file_name().context("No filename")?.to_string_lossy();

                if !dir_name.contains("_backup") {
                    let backup_dir = path.with_file_name(format!("{dir_name}_wav_backup"));
                    std::fs::create_dir_all(&backup_dir)?;

                    std::fs::read_dir(&path)?
                        .into_iter()
                        .par_bridge()
                        .flatten()
                        .filter(|e| {
                            e.file_type().map_or(false, |f| f.is_file())
                                && e.path().extension().map_or(false, |ext| ext == "wav")
                        })
                        .try_for_each(|wav_entry| {
                            let wav_path = wav_entry.path();
                            let mut wav_file = wavers::Wav::<f32>::from_path(&wav_path)?;
                            let audio_data = st_system::postprocessing::AudioData::new(&mut wav_file)?;

                            audio_data.write_to_ogg_vorbis(&wav_path.with_extension("ogg"), 0.6)?;

                            let backup_wav = wav_path.file_name().expect("Impossible");
                            std::fs::rename(&wav_path, backup_dir.join(backup_wav))?;
                            Ok::<_, eyre::Error>(())
                        })?;
                }
            }
        }

        Ok(())
    }
}
