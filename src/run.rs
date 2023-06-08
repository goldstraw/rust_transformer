use crate::embedding::load_embeddings;

pub fn run() {
    let word_embeddings = load_embeddings("word_embeddings.json");
}