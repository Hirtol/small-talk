use burn::backend::{ndarray::NdArrayDevice, NdArray};
use small_talk_ml::emotion_classifier::BasicEmotionClassifier;

mod f5_rs;
mod audio;
mod whisper;
mod pytests;

type Back = NdArray<f32, i32>;

fn main() -> eyre::Result<()> {
    // let device = NdArrayDevice::default();
    // let mut classifier: BasicEmotionClassifier<Back> = BasicEmotionClassifier::new("models/text_emotion_classifier/classifier_head", "models/text_emotion_classifier/ggml-model-Q4_k.gguf", device).unwrap();
    // whisper::main()
    let out = pytests::main();

    println!("OUT: {:?}", out);

    out
}
