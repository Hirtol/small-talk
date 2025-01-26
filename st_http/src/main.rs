use tracing_subscriber::util::SubscriberInitExt;
use st_http::setup::Application;
use st_http::{get_quit_notifier, telemetry};

#[tokio::main]
async fn main() -> eyre::Result<()> {
    // We don't care if it can't find a .env file
    let _ = dotenv::dotenv();

    color_eyre::install()?;

    // Setup Tracing
    let subscriber = telemetry::create_subscriber(
        "WARN,reqwest=DEBUG,st_system=TRACE,st_http=TRACE,st_ml=TRACE,sqlx=WARN,hyper=WARN",
    );
    subscriber.init();

    // Setup server
    let config = st_http::config::initialise_config()?;
    let app = Application::new(config).await?;

    let notifier = get_quit_notifier();

    app.run(notifier).await?;
    
    Ok(())
}
