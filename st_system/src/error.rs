use tokio::time::error::Elapsed;

error_set::error_set! {
    TtsError = GameSessionError;

    GameSessionError = {
        #[display("A line was incorrectly generated")]
        IncorrectGeneration,
    } || VoiceManagerError || RvcError;
    VoiceManagerError = {
        #[display("Requested voice: '{voice}' does not exist")]
        VoiceDoesNotExist {
            voice: String,
        }
    };
    RvcError = {
        #[display("Generation timeout, perhaps you are using a model that is too big")]
        Timeout(Elapsed),
    } || EyreError;
    EyreError = {
        #[display("Internal error, please submit a bug report: {0}")]
        Other(eyre::Error)
    };
}