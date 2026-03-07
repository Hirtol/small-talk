use std::sync::Arc;
use clap::Parser;
use tracing_subscriber::util::SubscriberInitExt;
use st_application::config::SmallTalkConfig;
use crate::args::SubCommands;

mod args;
mod trace;

#[derive(serde::Serialize, serde::Deserialize, Clone, Debug, Default)]
pub struct SmallTalkOrganiserConfig {
    #[serde(default)]
    pub small_talk: SmallTalkConfig,
}

#[tokio::main]
async fn main() -> eyre::Result<()> {
    color_eyre::install()?;
    let args = args::ClapArgs::parse();
    let conf = Arc::new(st_application::config::initialise_config::<SmallTalkOrganiserConfig>()?.small_talk);
    trace::create_subscriber("ERROR,st_http=TRACE,st_organiser=TRACE,st_ml=TRACE,st_system=TRACE").init();

    let now = std::time::Instant::now();

    match args.commands {
        SubCommands::Organise(solv) => {
            solv.run(conf).await?;
        }
        SubCommands::Compress(comp) => {
            comp.run(conf).await?;
        }
        SubCommands::ReassignVoice(reas) => {
            reas.run(conf).await?;
        }
        SubCommands::RegenerateLines(re) => {
            re.run(conf).await?;
        }
    }

    tracing::info!(
        "Runtime: {:.2?}s", now.elapsed().as_secs()
    );
    
    Ok(())
}
