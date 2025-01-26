use indicatif::{ProgressState, ProgressStyle};
use std::{fmt, time::Duration};
use tracing::Subscriber;
use tracing_indicatif::IndicatifLayer;
use tracing_subscriber::{
    fmt::{format::Writer, time::FormatTime},
    layer::SubscriberExt,
    EnvFilter, Layer,
};

pub fn create_subscriber(default_directives: &str) -> impl Subscriber {
    let env_filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(default_directives));
    let indicatif_layer = IndicatifLayer::new()
        .with_progress_style(
            ProgressStyle::with_template(
                "{color_start}{span_child_prefix}{spinner} {span_name} {wide_msg} {elapsed_subsec}{color_end}",
            )
            .unwrap()
            .with_key("elapsed_subsec", elapsed_subsec)
            .with_key(
                "color_start",
                |state: &ProgressState, writer: &mut dyn std::fmt::Write| {
                    let elapsed = state.elapsed();

                    if elapsed > Duration::from_secs(8) {
                        // Red
                        let _ = write!(writer, "\x1b[{}m", 1 + 30);
                    } else if elapsed > Duration::from_secs(4) {
                        // Yellow
                        let _ = write!(writer, "\x1b[{}m", 3 + 30);
                    }
                },
            )
            .with_key(
                "color_end",
                |state: &ProgressState, writer: &mut dyn std::fmt::Write| {
                    if state.elapsed() > Duration::from_secs(4) {
                        let _ = write!(writer, "\x1b[0m");
                    }
                },
            ),
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
                .with_writer(indicatif_layer.get_stderr_writer())
                .event_format(format)
                .with_filter(env_filter),
        )
        .with(indicatif_layer)
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

fn elapsed_subsec(state: &ProgressState, writer: &mut dyn std::fmt::Write) {
    let seconds = state.elapsed().as_secs();
    let sub_seconds = (state.elapsed().as_millis() % 1000) / 100;
    let _ = writer.write_str(&format!("{}.{}s", seconds, sub_seconds));
}
