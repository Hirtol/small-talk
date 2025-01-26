use burn::data::dataloader::batcher::Batcher;
use burn::prelude::{Backend, Int, Tensor, TensorData};

#[derive(Clone)]
pub struct EmotionBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> EmotionBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self {
            device,
        }
    }
}

#[derive(Debug, Clone)]
pub struct EmotionItem {
    pub text_embedding: Vec<f32>,
    pub label: usize,
}

#[derive(Clone, Debug)]
pub struct EmotionTrainingBatch<B: Backend> {
    pub embeddings: Tensor<B, 2>,
    pub labels: Tensor<B, 1, Int>,
}

#[derive(Clone, Debug)]
pub struct EmotionInferBatch<B: Backend> {
    /// Last layer of whatever underlying embedding model we're using.
    pub embeddings: Tensor<B, 2>,
}

impl<B: Backend> Batcher<EmotionItem, EmotionTrainingBatch<B>> for EmotionBatcher<B> {
    fn batch(&self, items: Vec<EmotionItem>) -> EmotionTrainingBatch<B> {
        let targets = items
            .iter()
            .map(|item| Tensor::<B, 1, Int>::from_data([item.label], &self.device))
            .collect();
        
        let emb_shape = items[0].text_embedding.len();
        let emb_tensor = items
            .into_iter()
            .map(|emb| TensorData::new(emb.text_embedding, [emb_shape]).convert::<B::FloatElem>())
            .map(|data| Tensor::<B, 1>::from_data(data, &self.device))
            .map(|tens| tens.unsqueeze_dim(0))
            .collect();

        let embeddings = Tensor::cat(emb_tensor, 0).to_device(&self.device);
        let labels = Tensor::cat(targets, 0).to_device(&self.device);

        EmotionTrainingBatch { embeddings, labels }
    }
}

impl<B: Backend> Batcher<Vec<f32>, EmotionInferBatch<B>> for EmotionBatcher<B> {
    fn batch(&self, items: Vec<Vec<f32>>) -> EmotionInferBatch<B> {
        let emb_shape = items[0].len();
        let emb_tensor = items
            .into_iter()
            .map(|emb| TensorData::new(emb, [emb_shape]).convert::<B::FloatElem>())
            .map(|data| Tensor::<B, 1>::from_data(data, &self.device))
            .map(|tens| tens.unsqueeze_dim(0))
            .collect();

        let embeddings = Tensor::cat(emb_tensor, 0).to_device(&self.device);

        EmotionInferBatch { embeddings }
    }
}