use std::sync::Arc;
use clap::Parser;
use tracing_subscriber::util::SubscriberInitExt;
use crate::args::SubCommands;

mod args;
mod trace;

#[tokio::main]
async fn main() -> eyre::Result<()> {
    color_eyre::install()?;
    let args = args::ClapArgs::parse();
    let conf = Arc::new(st_http::config::initialise_config()?);
    trace::create_subscriber("ERROR,st_http=TRACE,st_organiser=TRACE,st_ml=TRACE").init();

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
    }

    tracing::info!(
        "Runtime: {:.2?}s", now.elapsed().as_secs()
    );
    
    Ok(())
}
