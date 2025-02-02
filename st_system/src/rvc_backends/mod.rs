use std::path::PathBuf;
use std::time::Duration;
use crate::error::{RvcError};
use crate::postprocessing::AudioData;
use crate::rvc_backends::seedvc::local::LocalSeedHandle;

pub mod seedvc;

/// The collection of RVC backend handles.
#[derive(Clone)]
pub struct RvcCoordinator {
    seed_vc: Option<LocalSeedHandle>,
    seed_vc_hq: Option<LocalSeedHandle>,
}

impl RvcCoordinator {
    pub fn new(seed_vc: Option<LocalSeedHandle>, seed_vc_hq: Option<LocalSeedHandle>) -> Self {
        Self {
            seed_vc,
            seed_vc_hq,
        }
    }

    pub async fn prepare_instance(&self, hq: bool) -> Result<(), RvcError> {
        if hq {
            let Some(seed_vc_hq) = self.seed_vc_hq.as_ref() else {
                return Err(RvcError::RvcNotInitialised)
            };
            Ok(seed_vc_hq.start_instance().await?)
        } else {
            let Some(seed_vc) = self.seed_vc.as_ref() else {
                return Err(RvcError::RvcNotInitialised)
            };
            Ok(seed_vc.start_instance().await?)
        }
    }

    /// Submit the given `req` to a RVC model.
    ///
    /// If `high_quality` was set the request will take longer, but it will result in a better quality result.
    #[tracing::instrument(skip(self))]
    pub async fn rvc_request(&self, req: BackendRvcRequest, high_quality: bool) -> Result<BackendRvcResponse, RvcError> {
        if high_quality {
            let Some(seed_vc_hq) = self.seed_vc_hq.as_ref() else {
                return Err(RvcError::RvcNotInitialised)
            };
            Ok(tokio::time::timeout(Duration::from_secs(40), seed_vc_hq.rvc_request(req)).await??)
        } else {
            let Some(seed_vc) = self.seed_vc.as_ref() else {
                return Err(RvcError::RvcNotInitialised)
            };
            Ok(tokio::time::timeout(Duration::from_secs(40), seed_vc.rvc_request(req)).await??)
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