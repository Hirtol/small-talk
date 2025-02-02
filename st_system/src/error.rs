use tokio::time::error::Elapsed;
use crate::TtsModel;

pub type Result<T> = std::result::Result<T, GameSessionError>;

error_set::error_set! {
    GameSessionError = {
        #[display("A line was incorrectly generated")]
        IncorrectGeneration,
        #[display("The given text contained invalid characters for TTS: {txt}")]
        InvalidText {
            txt: String,
        },
    } || VoiceManagerError || RvcError || EmotionError || TtsError;

    VoiceManagerError = {
        #[display("Requested voice: '{voice}' does not exist")]
        VoiceDoesNotExist {
            voice: String,
        },
        #[display("Requested voice: '{voice}' has a directory, but no voice samples exist")]
        NoVoiceSamples {
            voice: String,
        }
    };

    TtsError = {
        #[display("The given TTS model does not have an active provider: {model:?}")]
        ModelNotInitialised {
            model: TtsModel,
        },
    } || EyreError;

    EmotionError = {
        #[display("Error loading model: {0}")]
        LoadError(st_ml::emotion_classifier::LoadError),
    } || EyreError;

    RvcError = {
        #[display("Generation timeout, perhaps you are using a model that is too big")]
        Timeout,
        #[display("No RVC model was given in the config, or was not available")]
        RvcNotInitialised
    } || EyreError;

    EyreError = {
        #[display("Internal error, please submit a bug report: {0}")]
        Other(eyre::Error)
    };
}

impl From<Elapsed> for RvcError {
    fn from(_: Elapsed) -> Self {
        RvcError::Timeout
    }
}