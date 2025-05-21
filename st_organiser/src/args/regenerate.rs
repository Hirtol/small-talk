use crate::args::ClapTtsModel;
use st_http::config::SharedConfig;

#[derive(clap::Args, Debug)]
pub struct RegenerateCommand {
    /// The name of the game-session which contains the voice we want to change.
    game_name: String,
    /// The voice to change
    voice: String,
    /// The location, either 'global' or '{GAME_NAME}'
    voice_location: String,
    #[clap(long)]
    model: ClapTtsModel,
}

impl RegenerateCommand {
    #[tracing::instrument(skip_all, fields(self.sample_path))]
    pub async fn run(self, config: SharedConfig) -> eyre::Result<()> {
        // Just piggyback off an existing command :)
        let command = super::reassign::ReassignCommand {
            game_name: self.game_name,
            voice: self.voice.clone(),
            voice_location: self.voice_location.clone(),
            target_voice: self.voice,
            target_location: self.voice_location,
            model: self.model,
        };
        command.run(config).await
    }
}
