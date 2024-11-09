use burn::data::dataset::transform::Mapper;
use crate::emotion_classifier::data::EmotionItem;
use crate::emotion_classifier::training::GoEmotionItem;

pub struct IntoEmotionItemMapper;

impl Mapper<GoEmotionItem, EmotionItem> for IntoEmotionItemMapper {
    fn map(&self, item: &GoEmotionItem) -> EmotionItem {
        todo!()
    }
}