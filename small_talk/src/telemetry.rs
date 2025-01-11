use tracing::Subscriber;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::{fmt, EnvFilter, Layer};

/// Create the initial subscriber, alongside the custom formatting for standard i/o.
pub fn create_subscriber(default_directives: &str) -> impl Subscriber + Send + Sync {
    let env_filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(default_directives));
    let rp_filter = tracing_subscriber::filter::filter_fn(|f| f.target().contains("small_"));

    let rp_format = tracing_subscriber::fmt::format()
        .with_level(true)
        .with_thread_ids(true)
        .with_source_location(true);
    let normal_format = rp_format.clone().with_source_location(false);

    // We only want file locations in small_* logs, we therefore filter those out in the normal_logger.
    let rp_logger = tracing_subscriber::fmt::layer()
        .event_format(rp_format)
        .with_filter(rp_filter)
        .with_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(default_directives)));
    let normal_logger = tracing_subscriber::fmt::layer()
        .event_format(normal_format)
        .with_filter(tracing_subscriber::filter::filter_fn(|m| !m.target().contains("small_")))
        .with_filter(env_filter);

    tracing_subscriber::registry().with(rp_logger).with(normal_logger)
}
