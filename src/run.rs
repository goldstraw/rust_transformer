use crate::block::Block;
use ndarray::arr2;
use crate::embedding::load_embeddings;
use crate::self_attention::SelfAttention;

pub fn run() {
    let _word_embeddings = load_embeddings("word_embeddings.json");
    let mut sa = SelfAttention::new(3, 3);
    sa.set_block(arr2(&[[1.040464,-0.99978536,-0.34397972], [-2.2353952,-1.1311471,-1.0246618], [2.797164,-2.737051,-0.5218795]]));
    sa.forward_propagate();
}