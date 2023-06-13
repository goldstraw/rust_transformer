use crate::block::Block;
use ndarray::arr1;
use crate::embedding::load_embeddings;
use crate::transformer::Transformer;

pub fn run() {
    let word_embeddings = load_embeddings("word_embeddings.json");
    let mut e = Transformer::new(3, 200, 3, arr1(&[600,400,600]), word_embeddings);
    e.forward_propagate(arr1(&["this".to_string(), "movie".to_string(), "was".to_string(), "absolutely".to_string(), "terrible".to_string()]));
    e.back_propagate(0.0);
}