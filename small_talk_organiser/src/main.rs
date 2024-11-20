use std::sync::Arc;
use clap::Parser;
use tracing_subscriber::util::SubscriberInitExt;
use crate::args::SubCommands;

mod args;
mod trace;

#[tokio::main]
async fn main() -> eyre::Result<()> {
    let args = args::ClapArgs::parse();
    let conf = Arc::new(small_talk::config::initialise_config()?);
    trace::create_subscriber("ERROR,small_talk=TRACE,small_talk_organiser=TRACE,small_talk_ml=TRACE").init();

    let now = std::time::Instant::now();

    match args.commands {
        SubCommands::Organise(solv) => {
            solv.run(conf).await?;
        }
    }

    tracing::info!(
        "Runtime: {:.2?}s", now.elapsed().as_secs()
    );
    
    Ok(())
}
