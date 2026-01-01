use std::fs::OpenOptions;
use tracing::Subscriber;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::{EnvFilter, Layer};
use tracing_subscriber::field::MakeExt;

/// Create the initial subscriber, alongside the custom formatting for standard i/o.
pub fn create_subscriber(default_directives: &str, file_target: String) -> impl Subscriber + Send + Sync {
    let env_filter = || EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(default_directives));
    let our_filter = tracing_subscriber::filter::filter_fn(|f| f.target().contains("st_"));

    let our_format = tracing_subscriber::fmt::format()
        .with_ansi(false)
        .with_target(true)
        .compact()
        .with_level(true)
        .with_thread_ids(true)
        .with_source_location(true);
    let normal_format = our_format.clone().with_source_location(false);

    // We only want file locations in small_* logs, we therefore filter those out in the normal_logger.
    let log_file = OpenOptions::new()
        .create(true)
        .write(true)
        .open(file_target)
        .unwrap();
    let our_logger = tracing_subscriber::fmt::layer()
        .event_format(our_format).map_fmt_fields(|f| f.debug_alt())
        .with_writer(log_file)
        .with_filter(our_filter);
    let normal_logger = tracing_subscriber::fmt::layer()
        .event_format(normal_format)
        .with_filter(tracing_subscriber::filter::filter_fn(|m| !m.target().contains("st_")));

    let subscriber = tracing_subscriber::registry()
        .with(env_filter())
        .with(our_logger)
        .with(normal_logger);

    subscriber
}