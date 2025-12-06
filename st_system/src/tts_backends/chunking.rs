use itertools::Itertools;
use regex::Regex;
use std::sync::LazyLock;

static SENTENCE_SPLIT_REGEX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"([.!?]+[\s]+)").unwrap());
static COMMA_SPLIT_REGEX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(,[\s]+)").unwrap());

#[derive(Debug, Clone)]
pub struct Chunk {
    pub text: String,
    pub word_count: usize,
    pub cost: usize,
}

fn estimate_word_count(text: &str) -> usize {
    text.split_whitespace().count()
}

fn calculate_chunk_cost(text: &str) -> usize {
    text.chars()
        .map(|c| match c {
            '.' => 5,
            ',' => 3,
            '-' => 3,
            ';' => 3,
            _ => 1,
        })
        .sum()
}

pub fn chunk_text(
    text: &str,
    max_words_size: usize,
    max_chunk_cost: usize,
) -> Vec<Chunk> {
    let total_words = estimate_word_count(text);
    let total_cost = calculate_chunk_cost(text);

    // Fits in a single chunk
    if total_words <= max_words_size && total_cost <= max_chunk_cost {
        return vec![Chunk {
            text: text.to_string(),
            word_count: total_words,
            cost: total_cost,
        }];
    }

    let sentences: Vec<&str> = text.split_inclusive(&*SENTENCE_SPLIT_REGEX)
        .filter(|s| !s.trim().is_empty())
        .collect();

    let mut chunks = Vec::<Chunk>::new();
    let mut current = String::new();
    let mut curr_words = 0usize;
    let mut curr_cost = 0usize;

    for sent in sentences {
        let wc = estimate_word_count(sent);
        let cst = calculate_chunk_cost(sent);

        // Splitting sentence
        if wc > max_words_size || cst > max_chunk_cost {
            // flush current
            if !current.trim().is_empty() {
                chunks.push(Chunk {
                    text: current.trim().to_string(),
                    word_count: curr_words,
                    cost: curr_cost,
                });
                current.clear();
                curr_words = 0;
                curr_cost = 0;
            }

            // Recurse
            chunks.extend(split_long_sentence(sent, max_words_size, max_chunk_cost));
            continue;
        }

        // Appends to current if safe
        if (curr_words + wc <= max_words_size)
            && (curr_cost + cst <= max_chunk_cost)
        {
            current.push_str(sent);
            curr_words += wc;
            curr_cost += cst;
        } else {
            // Emit current
            if !current.trim().is_empty() {
                chunks.push(Chunk {
                    text: current.trim().to_string(),
                    word_count: curr_words,
                    cost: curr_cost,
                });
            }

            current = sent.to_string();
            curr_words = wc;
            curr_cost = cst;
        }
    }

    if !current.trim().is_empty() {
        chunks.push(Chunk {
            text: current.trim().to_string(),
            word_count: curr_words,
            cost: curr_cost,
        });
    }

    chunks
}

fn split_long_sentence(
    text: &str,
    max_words_size: usize,
    max_chunk_cost: usize,
) -> Vec<Chunk> {
    let parts: Vec<&str> = text.split_inclusive(&*COMMA_SPLIT_REGEX)
        .filter(|s| !s.trim().is_empty())
        .collect();

    if parts.len() == 1 {
        return split_by_word_cut(text, max_words_size);
    }

    let mut chunks = Vec::<Chunk>::new();
    let mut current = String::new();
    let mut curr_words = 0;
    let mut curr_cost = 0;

    for p in parts {
        let wc = estimate_word_count(p);
        let cst = calculate_chunk_cost(p);

        if wc > max_words_size || cst > max_chunk_cost {
            if !current.trim().is_empty() {
                chunks.push(Chunk {
                    text: current.trim().to_string(),
                    word_count: curr_words,
                    cost: curr_cost,
                });
                current.clear();
                curr_words = 0;
                curr_cost = 0;
            }
            chunks.extend(split_by_word_cut(p, max_words_size));
            continue;
        }

        if curr_words + wc <= max_words_size && curr_cost + cst <= max_chunk_cost {
            current.push_str(p);
            curr_words += wc;
            curr_cost += cst;
        } else {
            chunks.push(Chunk {
                text: current.trim().to_string(),
                word_count: curr_words,
                cost: curr_cost,
            });
            current = p.to_string();
            curr_words = wc;
            curr_cost = cst;
        }
    }

    if !current.trim().is_empty() {
        chunks.push(Chunk {
            text: current.trim().to_string(),
            word_count: curr_words,
            cost: curr_cost,
        });
    }

    chunks
}

/// Final fallback: slice by N words and build Chunk info
fn split_by_word_cut(text: &str, max_words: usize) -> Vec<Chunk> {
    let mut chunks = Vec::<Chunk>::new();
    let mut buf = Vec::<&str>::new();
    let mut count = 0;

    for w in text.split_whitespace() {
        if count >= max_words {
            let s = buf.join(" ");
            chunks.push(Chunk {
                cost: calculate_chunk_cost(&s),
                word_count: buf.len(),
                text: s,
            });
            buf.clear();
            count = 0;
        }
        buf.push(w);
        count += 1;
    }

    if !buf.is_empty() {
        let s = buf.join(" ");
        chunks.push(Chunk {
            cost: calculate_chunk_cost(&s),
            word_count: buf.len(),
            text: s,
        });
    }

    chunks
}

#[cfg(test)]
mod tests {
    use crate::tts_backends::chunking::{chunk_text, SENTENCE_SPLIT_REGEX};

    #[test]
    fn test_chunk_calculate() {
        let test_sentence = r#"
        "On another day, I would argue with you, priest.
            I would say a hunch is a poor substitute for a plan, that intuition won't feed you when the coffers are empty,
            and that spontaneity won't warm you if you've not chopped firewood in advance.
            But now... These city folk had planned for every possibility, they thought they were ready for everything,
            and yet all it took was the snap of a finger for their world to come tumblin' down."
            "#;

        // let out = SENTENCE_SPLIT_REGEX.split(test_sentence).collect::<Vec<&str>>();
        let re = &*SENTENCE_SPLIT_REGEX;
        let out = chunk_text(test_sentence, 70, 500);
        println!("{:#?}", out);
    }
}
