use std::{fmt, time::Duration};
use indicatif::ProgressState;
use tracing::Subscriber;
use tracing_indicatif::filter::{hide_indicatif_span_fields, IndicatifFilter};
use tracing_indicatif::IndicatifLayer;
use tracing_indicatif::style::ProgressStyle;
use tracing_subscriber::{
    fmt::{format::Writer, time::FormatTime},
    layer::SubscriberExt,
    EnvFilter, Layer,
};
use tracing_subscriber::fmt::format::DefaultFields;

pub fn create_subscriber(default_directives: &str) -> impl Subscriber {
    let env_filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(default_directives));
    let indicatif_layer = IndicatifLayer::new()
        .with_progress_style(
            ProgressStyle::with_template(
                "{span_child_prefix}{spinner} {span_name} {wide_msg} {elapsed_precise}",
            )
            .unwrap(),
        )
        .with_span_child_prefix_symbol("↳ ")
        .with_span_child_prefix_indent(" ");

    let format = tracing_subscriber::fmt::format()
        .with_source_location(false)
        .with_file(false)
        .with_timer(Uptime::default());

    tracing_subscriber::registry()
        .with(
            tracing_subscriber::fmt::layer()
                .event_format(format)
                .with_writer(indicatif_layer.get_stderr_writer())
                .with_filter(env_filter),
        )
        .with(indicatif_layer.with_filter(tracing_subscriber::filter::filter_fn(|m| m.target().contains("st_organiser"))))
}

struct Uptime(std::time::Instant);

impl Default for Uptime {
    fn default() -> Self {
        Uptime(std::time::Instant::now())
    }
}

impl FormatTime for Uptime {
    fn format_time(&self, w: &mut Writer<'_>) -> fmt::Result {
        let e = self.0.elapsed();
        let sub_seconds = (e.as_millis() % 1000) / 100;
        write!(w, "{}.{}s", e.as_secs(), sub_seconds)
    }
}
