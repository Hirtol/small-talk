use crate::args::organise::OrganiseCommand;

pub mod organise;
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
}