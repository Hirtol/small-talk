use std::path::Path;
use eyre::Context;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::token::LlamaToken;

pub static LLAMA_BACKEND: std::sync::LazyLock<std::sync::Mutex<LlamaBackend>> = std::sync::LazyLock::new(|| std::sync::Mutex::new(LlamaBackend::init().unwrap()));

self_cell::self_cell!(
    pub struct LlamaCell {
        owner: LlamaModel,
        #[covariant]
        dependent: LlamaContext,
    }
);

pub struct LLamaEmbedder {
    model: LlamaCell,
    batch: LlamaBatch,
}

// SAFETY: GGML/LLama.CPP seem to be thread-safe since the start of this year (2024), as in, no thread-local state to worry about
// (see https://github.com/ggerganov/llama.cpp/discussions/499). 
unsafe impl Send for LLamaEmbedder {}

impl LLamaEmbedder {
    /// Create a [LLamaEmbedder] by initialising the `LLama.CPP` backend and loading the given model from `model_path`.
    ///
    /// # Arguments
    /// * `model_path` - The location of the GGUF model to be loaded
    /// * `model_params` - The parameters for the `LLama.CPP` model
    /// * `ctx_params` - The parameters for the context
    /// * `n_batch` - If one wants to forcefully limit the amount of tokens then provide [Some], otherwise just leave it to [None]
    /// to use the default `n_ctx` of the model as the max.
    pub fn new(
        model_path: impl AsRef<Path>,
        model_params: LlamaModelParams,
        ctx_params: LlamaContextParams,
        n_batch: Option<u32>,
    ) -> eyre::Result<Self> {
        let mut backend = LLAMA_BACKEND.lock()?;
        backend.void_logs();

        let model = LlamaModel::load_from_file(backend, model_path, &model_params)?;

        let ctx_params = ctx_params.with_embeddings(true);
        let ctx_params = if let Some(n_batch) = n_batch {
            ctx_params.with_n_batch(n_batch)
        } else {
            ctx_params
        };
        let cell = LlamaCell::try_new(model, |model| {
            model
                .new_context(backend, ctx_params)
                .context("unable to create the llama_context")
        })?;

        let n_ctx = cell.borrow_dependent().n_ctx();
        let n_train = cell.borrow_owner().n_ctx_train();
        if n_train < n_ctx {
            eyre::bail!("Given context's `n_ctx` (`{n_ctx}`) was larger than the model was trained for (`{n_train}`).")
        }

        let batch = LlamaBatch::new(n_batch.unwrap_or(n_ctx) as usize, 1);

        Ok(Self {
            model: cell,
            batch,
        })
    }
    
    /// Create a new embedding context with sensible defaults.
    pub fn new_default(model_path: impl AsRef<Path>) -> eyre::Result<Self> {
        let model_params = LlamaModelParams::default().with_n_gpu_layers(0);
        let ctx_params = LlamaContextParams::default()
            .with_n_threads(16)
            .with_n_threads_batch(16)
            .with_n_ctx(None) // Load from model
            .with_n_batch(512)
            .with_embeddings(true);
        
        Self::new(model_path, model_params, ctx_params, None)
    }

    /// Tokenize the given texts using the default `LLama.CPP` tokenizer for the current model.
    pub fn tokenize(
        &self,
        texts: impl IntoIterator<Item = impl AsRef<str>>,
    ) -> eyre::Result<Vec<Vec<LlamaToken>>> {
        texts
            .into_iter()
            .map(|text| {
                self.model
                    .borrow_owner()
                    .str_to_token(text.as_ref(), AddBos::Always)
                    .context("Failed to tokenize")
            })
            .collect()
    }

    pub fn embed_tokens(
        &mut self,
        batch_tokens: &mut [Vec<LlamaToken>],
        normalise: bool,
        truncate: bool,
    ) -> eyre::Result<Vec<Vec<f32>>> {
        let mut output = Vec::with_capacity(batch_tokens.len());

        let mut max_seq_id = 0;
        let n_batch = self.model.borrow_dependent().n_batch() as usize;

        for tokens in batch_tokens {
            // Force the prompt to be at most the size of the context
            if truncate {
                tokens.truncate(n_batch);
            } else if tokens.len() > n_batch {
                eyre::bail!(
                    "The given tokens produces `{}` tokens when at most `{}` are allowed in a batch",
                    tokens.len(),
                    n_batch
                );
            }

            // Batch has been filled up
            if (self.batch.n_tokens() as usize + tokens.len()) > n_batch {
                self.decode_batch(&mut output, max_seq_id, normalise);
                max_seq_id = 0;
            }

            self.batch.add_sequence(tokens, max_seq_id as i32, false)?;
            max_seq_id += 1;
        }

        // Handle last batch
        self.decode_batch(&mut output, max_seq_id, normalise);

        Ok(output)
    }

    /// Create embeddings of the given text sequences in a batched form.
    ///
    /// # Arguments
    /// * `texts` - The texts to create embeddings for
    /// * `normal` - How the embeddings should be normalised
    /// * `truncate` - Whether to truncate any of the given texts if they produce more tokens than the context window of the model can handle.
    /// If this is not set to `true` then the function will return with an error instead.
    pub fn embed(
        &mut self,
        texts: impl IntoIterator<Item = impl AsRef<str>>,
        normalise: bool,
        truncate: bool,
    ) -> eyre::Result<Vec<Vec<f32>>> {
        let texts = texts.into_iter();
        let mut output = Vec::with_capacity(texts.size_hint().1.unwrap_or(0));

        let mut t_batch = 0;
        let mut p_batch = 0;
        let n_batch = self.model.borrow_dependent().n_batch() as usize;

        for text in texts {
            let mut tokens = self
                .model
                .borrow_owner()
                .str_to_token(text.as_ref(), AddBos::Always)?;

            // Force the prompt to be at most the size of the context
            if truncate {
                tokens.truncate(n_batch);
            } else if tokens.len() > n_batch {
                eyre::bail!(
                    "The given text: `{}` produces `{}` tokens when at most `{}` are allowed in a batch",
                    text.as_ref(),
                    tokens.len(),
                    n_batch
                );
            }

            let n_tokens = tokens.len();

            // Batch has been filled up
            if t_batch + n_tokens > n_batch {
                self.decode_batch(&mut output, p_batch, normalise);
                t_batch = 0;
                p_batch = 0;
            }

            self.batch.add_sequence(&tokens, p_batch as i32, false)?;
            t_batch += n_tokens;
            p_batch += 1;
        }

        // Handle last batch
        self.decode_batch(&mut output, p_batch, normalise);

        Ok(output)
    }

    fn decode_batch(&mut self, output: &mut Vec<Vec<f32>>, n_sequence: usize, normalise: bool) {
        self.model.with_dependent_mut(|_, ctx| {
            ctx.clear_kv_cache();
            let _ = ctx.decode(&mut self.batch);

            for i in 0..n_sequence {
                let embedding = ctx
                    .embeddings_seq_ith(i as i32)
                    .expect("Embeddings are always enabled");
                // Either normalise immediately, or perform mean-pooling.
                let embedding = if normalise {
                    normalize(embedding)
                } else {
                    embedding.to_vec()
                };

                output.push(embedding);
            }

            self.batch.clear();
        });
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, PartialOrd, Ord, Hash)]
pub enum Normalisation {
    /// Normalise the embeddings, ensuring an efficient cosine similarity can be computed
    Normalise,
    /// Only use mean pooling on the output embedding.
    None,
}

pub fn normalize(vec: &[f32]) -> Vec<f32> {
    let magnitude = (vec.iter().fold(0.0, |acc, &val| val.mul_add(val, acc))).sqrt();

    if magnitude > f32::EPSILON {
        vec.iter().map(|&val| val / magnitude).collect()
    } else {
        vec.to_vec()
    }
}