use tokio::time::error::Elapsed;

pub type Result<T> = std::result::Result<T, TtsError>;

error_set::error_set! {
    TtsError = GameSessionError;

    GameSessionError = {
        #[display("A line was incorrectly generated")]
        IncorrectGeneration,
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
        Timeout(Elapsed),
    } || EyreError;

    EyreError = {
        #[display("Internal error, please submit a bug report: {0}")]
        Other(eyre::Error)
    };
}