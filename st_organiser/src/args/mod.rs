use crate::args::compress::CompressCommand;
use crate::args::migrate::MigrateCommand;
use crate::args::organise::OrganiseCommand;
use crate::args::reassign::ReassignCommand;
use crate::args::regenerate::RegenerateCommand;

pub mod organise;
pub mod compress;
pub mod reassign;
pub mod regenerate;
pub mod migrate;

#[derive(clap::Parser, Debug)]
#[clap(version, about)]
pub struct ClapArgs {
    #[clap(subcommand)]
    pub commands: SubCommands,
}

#[derive(clap::Subcommand, Debug)]
pub enum SubCommands {
    /// Organise voice data into the directory structure expected by SmallTalk
    #[clap(arg_required_else_help(true))]
    #[clap(alias = "o")]
    Organise(OrganiseCommand),
    /// Compress the generated lines to OGG Vorbis
    #[clap(arg_required_else_help(true))]
    #[clap(alias = "c")]
    Compress(CompressCommand),
    /// Change the assignment of all characters with a given voice to a new voice, and regenerate their lines.
    #[clap(arg_required_else_help(true))]
    #[clap(alias = "c")]
    ReassignVoice(ReassignCommand),
    /// Regenerate all the lines of the given voice.
    #[clap(arg_required_else_help(true))]
    #[clap(alias = "c")]
    RegenerateLines(RegenerateCommand),
    #[clap(arg_required_else_help(true))]
    #[clap(alias = "c")]
    Migrate(MigrateCommand)
}

#[derive(clap::ValueEnum, Clone, Copy, Debug)]
pub enum ClapTtsModel {
    Xtts,
    IndexTts
}

impl From<ClapTtsModel> for st_system::TtsModel {
    fn from(value: ClapTtsModel) -> Self {
        match value {
            ClapTtsModel::Xtts => st_system::TtsModel::Xtts,
            ClapTtsModel::IndexTts => st_system::TtsModel::IndexTts
        }
    }
}