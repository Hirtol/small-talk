use tokio::time::error::Elapsed;

pub type Result<T> = std::result::Result<T, TtsError>;

error_set::error_set! {
    TtsError = GameSessionError;

    GameSessionError = {
        #[display("A line was incorrectly generated")]
        IncorrectGeneration,
        #[display("The given text contained invalid characters for TTS: {txt}")]
        InvalidText {
            txt: String,
        },
    } || VoiceManagerError || RvcError || EmotionError;

    VoiceManagerError = {
        #[display("Requested voice: '{voice}' does not exist")]
        VoiceDoesNotExist {
            voice: String,
        }
    };

    EmotionError = {
        #[display("Error loading model: {0}")]
        LoadError(st_ml::emotion_classifier::LoadError),
    } || EyreError;

    RvcError = {
        #[display("Generation timeout, perhaps you are using a model that is too big")]
        Timeout,
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