use std::time::Duration;
use eyre::{ContextCompat, OptionExt};

/// A simple cell which can automatically drop the contained state when it hasn't been accessed for a given `timeout`.
///
/// Expects [Self::timeout_future] to be awaited in a [tokio::select!] call.
pub struct GcCell<T> {
    timeout: Duration,
    last_access: std::time::Instant,
    state: Option<T>
}

impl<T: DroppableState> GcCell<T> {
    pub fn new(timeout: Duration) -> Self {
        Self {
            timeout,
            last_access: std::time::Instant::now(),
            state: None
        }
    }

    /// This future needs to be awaited in order to properly handle timeouts.
    ///
    /// It will not resolve until the `timeout` given in the constructor has elapsed *if* there is initialised state.
    /// If there is no initialised state it will simply never resolve.
    ///
    /// Best used in a `tokio::select!` macro, as it is cancel-safe.
    ///
    /// If it resolves the callee has to manually call [Self::kill_state]
    pub async fn timeout_future(&mut self) {
        if self.state.is_none() {
            std::future::pending().await
        } else {
            let timeout = self.last_access + self.timeout;
            tokio::time::sleep_until(timeout.into()).await;
        }
    }

    /// Get the state inside the [GcCell].
    ///
    /// If it hasn't been initialised, or if it has been dropped in the meantime, it will be re-initialised before returning.
    pub async fn get_state(&mut self, ctx: &T::Context) -> eyre::Result<&mut T> {
        // Borrow checker prevents us from doing this nicely...
        let out = if self.state.is_none() {
            let new_state = T::initialise_state(ctx).await?;
            self.state = Some(new_state);
            self.state.as_mut().ok_or_eyre("Impossible")
        } else {
            self.state.as_mut().context("Impossible")
        };

        self.last_access = std::time::Instant::now();

        out
    }

    /// Delete the current state.
    pub async fn kill_state(&mut self) -> eyre::Result<()> {
        let Some(mut val) = self.state.take() else {
            return Ok(());
        };
        val.on_kill().await?;
        Ok(())
    }
}

pub trait DroppableState: Sized {
    /// Context needed to properly initialise the state.
    type Context;

    /// Initialise this state using the provided [Self::Context]
    async fn initialise_state(context: &Self::Context) -> eyre::Result<Self>;

    /// Async drop for cleanup, will be called when the state is dropped
    async fn on_kill(&mut self) -> eyre::Result<()>;
}