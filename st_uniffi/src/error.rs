
// #[derive(uniffi::Error, Debug, Clone)]
// pub enum FfiError {
//     Unknown,
//     Details(String)
// }

error_set::error_set! {
    #[derive(uniffi::Error)]
    FfiError = {
        #[display("An unknown error occurred")]
        Unknown,
        #[display("An error occurred: {txt}")]
        Details {
            txt: String
        }
    };
}

impl From<eyre::Error> for FfiError {
    fn from(value: eyre::Error) -> Self {
        FfiError::Details {
            txt: value.to_string(),
        }
    }
}