use tracing_subscriber::util::SubscriberInitExt;
use small_talk::setup::Application;
use small_talk::{get_quit_notifier, telemetry};

#[tokio::main]
async fn main() -> eyre::Result<()> {
    // We don't care if it can't find a .env file
    let _ = dotenv::dotenv();

    color_eyre::install()?;

    // Setup Tracing
    let subscriber = telemetry::create_subscriber(
        "WARN,reqwest=DEBUG,small_talk=TRACE,small_talk_ml=TRACE,sqlx=WARN,hyper=WARN",
    );
    subscriber.init();

    // Setup server
    let config = small_talk::config::initialise_config()?;
    let app = Application::new(config).await?;

    let notifier = get_quit_notifier();

    app.run(notifier).await?;
    
    Ok(())
}
