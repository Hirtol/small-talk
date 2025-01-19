use std::path::PathBuf;
use std::time::Duration;
use crate::system::rvc_backends::seedvc::local::LocalSeedHandle;

pub mod seedvc;

/// The collection of RVC backend handles.
#[derive(Clone)]
pub struct RvcBackend {
    seed_vc: LocalSeedHandle,
}

impl RvcBackend {
    pub fn new(seed_vc: LocalSeedHandle) -> Self {
        Self {
            seed_vc,
        }
    }

    /// Submit the given `req` to a RVC model.
    #[tracing::instrument(skip(self))]
    pub async fn rvc_request(&self, req: BackendRvcRequest) -> eyre::Result<BackendRvcResponse> {
        self.seed_vc.rvc_request(req).await
    }
}

#[derive(Debug, Clone)]
pub struct BackendRvcRequest {
    samples: Vec<f32>,
    channels: u8,
    sample_rate: u32,
    target_voice: PathBuf,
}

#[derive(Debug)]
pub struct BackendRvcResponse {
    /// How long it took to generate the response
    pub gen_time: Duration,
    pub result: RvcResult
}

#[derive(Debug)]
pub enum RvcResult {
    /// FS location of the output
    Wav(wavers::Wav<f32>),
    /// TODO, maybe
    Stream
}
// SAFETY: wavers didn't specify `Send` in its internal implementation for its Box<dyn> type.
// This is a hack around that, as the types we provide it are Send, but this is _still_ undefined behavior.
unsafe impl Send for RvcResult {}

pub trait DroppableState {
    async fn ready(&self) -> eyre::Result<bool>;
}