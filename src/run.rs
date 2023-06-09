use crate::block::Block;
use ndarray::{arr1,arr2};
use crate::embedding::load_embeddings;
use crate::self_attention::SelfAttention;
use crate::multi_headed_attention::MultiHeadedAttention;
use crate::dense::Dense;

pub fn run() {
    let _word_embeddings = load_embeddings("word_embeddings.json");
    // let mut sa = SelfAttention::new(3, 3);
    // sa.set_block(arr2(&[[1.040464,-0.99978536,-0.34397972], [-2.2353952,-1.1311471,-1.0246618], [2.797164,-2.737051,-0.5218795]]));
    // sa.forward_propagate();

    // let mut d = Dense::new(arr1(&[10,8,5]),false);
    // d.set_block(arr1(&[1.040464,-0.99978536,-0.34397972, 1.040464,-0.99978536,-0.34397972, 1.040464,-0.99978536,-0.34397972, 1.040464]));
    // d.forward_propagate();

    let mut sa = MultiHeadedAttention::new(3, 3, 3);
    sa.set_block(arr2(&[[1.040464,-0.99978536,-0.34397972], [-2.2353952,-1.1311471,-1.0246618], [2.797164,-2.737051,-0.5218795]]));
    sa.forward_propagate();
}