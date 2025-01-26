use tracing::Subscriber;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::{fmt, EnvFilter, Layer};

/// Create the initial subscriber, alongside the custom formatting for standard i/o.
pub fn create_subscriber(default_directives: &str) -> impl Subscriber + Send + Sync {
    let env_filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(default_directives));
    let our_filter = tracing_subscriber::filter::filter_fn(|f| f.target().contains("small_") || f.target().contains("st_system"));

    let our_format = tracing_subscriber::fmt::format()
        .with_level(true)
        .with_thread_ids(true)
        .with_source_location(true);
    let normal_format = our_format.clone().with_source_location(false);

    // We only want file locations in small_* logs, we therefore filter those out in the normal_logger.
    let our_logger = tracing_subscriber::fmt::layer()
        .event_format(our_format)
        .with_filter(our_filter)
        .with_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(default_directives)));
    let normal_logger = tracing_subscriber::fmt::layer()
        .event_format(normal_format)
        .with_filter(tracing_subscriber::filter::filter_fn(|m| !m.target().contains("small_")))
        .with_filter(env_filter);

    let console = console_subscriber::spawn();

    tracing_subscriber::registry().with(our_logger).with(normal_logger).with(console)
}
