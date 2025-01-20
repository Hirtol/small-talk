use std::path::PathBuf;
use std::time::Duration;
use crate::system::postprocessing::AudioData;
use crate::system::rvc_backends::seedvc::local::LocalSeedHandle;

pub mod seedvc;

/// The collection of RVC backend handles.
#[derive(Clone)]
pub struct RvcBackend {
    seed_vc: LocalSeedHandle,
    seed_vc_hq: LocalSeedHandle,
}

impl RvcBackend {
    pub fn new(seed_vc: LocalSeedHandle, seed_vc_hq: LocalSeedHandle) -> Self {
        Self {
            seed_vc,
            seed_vc_hq,
        }
    }

    pub async fn prepare_instance(&self, hq: bool) -> eyre::Result<()> {
        if hq {
            self.seed_vc_hq.start_instance().await
        } else {
            self.seed_vc.start_instance().await
        }
    }

    /// Submit the given `req` to a RVC model.
    ///
    /// If `high_quality` was set the request will take longer, but it will result in a better quality result.
    #[tracing::instrument(skip(self))]
    pub async fn rvc_request(&self, req: BackendRvcRequest, high_quality: bool) -> eyre::Result<BackendRvcResponse> {
        if high_quality {
            self.seed_vc_hq.rvc_request(req).await
        } else {
            self.seed_vc_hq.rvc_request(req).await
        }
    }
}

#[derive(Debug, Clone)]
pub struct BackendRvcRequest {
    pub audio: AudioData,
    pub target_voice: PathBuf,
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
    Wav(AudioData),
    /// TODO, maybe
    Stream
}