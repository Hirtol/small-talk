use burn::config::Config;
use burn::module::Module;
use burn::nn::{Dropout, DropoutConfig, Linear, LinearConfig};
use burn::nn::loss::CrossEntropyLossConfig;
use burn::prelude::{Backend, Int, Tensor};
use burn::train::ClassificationOutput;
use crate::emotion_classifier::data::EmotionInferBatch;

#[derive(Module, Debug)]
pub struct EmotionModel<B: Backend> {
    classifier: Linear<B>,
    dropout: Dropout
}

#[derive(Config, Debug)]
pub struct EmotionModelConfig {
    /// Size of the incoming embedding vector.
    incoming_features: usize,
    /// Number of output classes
    num_classes: usize,
    #[config(default = "0.5")]
    dropout: f64,
}

impl EmotionModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> EmotionModel<B> {
        EmotionModel {
            classifier: LinearConfig::new(self.incoming_features, self.num_classes).init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
}

impl<B: Backend> EmotionModel<B> {
    /// # Shapes
    ///   - Embedding [batch_size, width]
    ///   - Output [batch_size, num_classes]
    pub fn forward(&self, embedding: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = embedding;
        // Drop part of the embeddings
        let x = self.dropout.forward(x);

        self.classifier.forward(x)
    }

    pub fn forward_classification(&self, embedding: Tensor<B, 2>, targets: Tensor<B, 1, Int>) -> ClassificationOutput<B> {
        let logits = self.forward(embedding);
        let loss = CrossEntropyLossConfig::new()
            .init(&logits.device())
            .forward(logits.clone(), targets.clone());

        ClassificationOutput::new(loss, logits, targets)
    }

    pub fn infer(&self, batch: EmotionInferBatch<B>) -> Tensor<B, 2> {
        // Dropout is automatically skipped if we have a back-end without autodiff, so can just call forward
        self.forward(batch.embeddings)
    }
}