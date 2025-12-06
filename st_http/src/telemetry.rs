use tracing::Subscriber;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::{fmt, EnvFilter, Layer};

/// Create the initial subscriber, alongside the custom formatting for standard i/o.
pub fn create_subscriber(default_directives: &str) -> impl Subscriber + Send + Sync {
    let env_filter = || EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(default_directives));
    let our_filter = tracing_subscriber::filter::filter_fn(|f| f.target().contains("st_"));

    let our_format = tracing_subscriber::fmt::format()
        .with_ansi(true)
        .with_target(true)
        .compact()
        .with_level(true)
        .with_thread_ids(true)
        .with_source_location(true);
    let normal_format = our_format.clone().with_source_location(false);

    // We only want file locations in small_* logs, we therefore filter those out in the normal_logger.
    // let our_logger = tracing_tree::HierarchicalLayer::new(2)
    //     .with_filter(our_filter)
    //     .with_filter(env_filter());
    let our_logger = tracing_subscriber::fmt::layer()
        .event_format(our_format)
        .with_filter(our_filter);
    let normal_logger = tracing_subscriber::fmt::layer()
        .event_format(normal_format)
        .with_filter(tracing_subscriber::filter::filter_fn(|m| !m.target().contains("st_")));

    let subscriber = tracing_subscriber::registry()
        .with(env_filter())
        .with(our_logger)
        .with(normal_logger);

    #[cfg(feature = "debug")]
    let subscriber = {
        let console = console_subscriber::spawn();
        subscriber.with(console)
    };

    #[cfg(feature = "venator")]
    let subscriber = {
        let vena = venator::Venator::default();
        subscriber.with(vena)
    };

    subscriber
}

// fn setup_tracing() {
//     // Parse log level from RUST_LOG (default: info)
//     let env_filter = EnvFilter::try_from_default_env()
//         .or_else(|_| EnvFilter::try_new("info"))
//         .unwrap();
//
//     // Human-readable layer for console (development only)
//     let stdout_log = fmt::layer()
//         .compact()  // Or .pretty() for multi-line
//         .with_ansi(true)
//         .with_target(true)
//         .with_writer(std::io::stdout)
//         .with_filter(tracing_subscriber::filter::LevelFilter::DEBUG);
//
//     // Structured JSON layer (file or stdout)
//     let (json_log, _guard) = match std::env::var("LOG_JSON").as_deref() {
//         Ok("stdout") => {
//             let (writer, guard) = non_blocking(std::io::stdout());
//             let layer = fmt::layer()
//                 .json()
//                 .flatten_event(true)  // Merge span fields into event
//                 .with_writer(writer);
//             (Some(layer), Some(guard))
//         }
//         Ok("file") => {
//             let file_appender = tracing_appender::rolling::daily("logs", "app.json");
//             let (writer, guard) = non_blocking(file_appender);
//             let layer = fmt::layer()
//                 .json()
//                 .flatten_event(true)
//                 .with_writer(writer);
//             (Some(layer), Some(guard))
//         }
//         _ => (None, None),
//     };
//
//     // Register layers
//     tracing_subscriber::registry()
//         .with(env_filter)
//         .with(stdout_log)  // Enabled in dev; omit in prod
//         .with(json_log)    // Conditionally enabled
//         .init();
// }