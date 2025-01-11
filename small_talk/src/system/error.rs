error_set::error_set! {
    TtsError = GameSessionError;

    GameSessionError = {
        #[display("Internal error, please submit a bug report: {0}")]
        Other(eyre::Error),
    } || VoiceManagerError;
    VoiceManagerError = {
        #[display("Requested voice: '{voice}' does not exist")]
        VoiceDoesNotExist {
            voice: String,
        }
    };
}