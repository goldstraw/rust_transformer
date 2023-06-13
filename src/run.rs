use crate::block::Block;
use ndarray::arr1;
use crate::embedding::load_embeddings;
use crate::transformer::Transformer;

pub fn run() {
    let word_embeddings = load_embeddings("word_embeddings.json");
    let mut e = Transformer::new(3, 3, 3, arr1(&[9,400,9]), word_embeddings);
    e.forward_propagate(arr1(&["movie".to_string(), "good".to_string(), "bad".to_string()]));
}