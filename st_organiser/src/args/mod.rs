use crate::args::compress::CompressCommand;
use crate::args::organise::OrganiseCommand;
use crate::args::reassign::ReassignCommand;

pub mod organise;
pub mod compress;
pub mod reassign;

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
    ReassignVoice(ReassignCommand)
}