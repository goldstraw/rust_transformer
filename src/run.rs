use crate::block::Block;
use crate::encoder_block::EncoderBlock;
use crate::positional_encoder::PositionalEncoder;
use ndarray::{arr1,arr2};
use crate::embedding::load_embeddings;
use crate::self_attention::SelfAttention;
use crate::multi_headed_attention::MultiHeadedAttention;
use crate::dense::Dense;
use crate::add_and_norm::AddAndNorm;
use crate::transformer::Transformer;

pub fn run() {
    let word_embeddings = load_embeddings("word_embeddings.json");
    // let mut sa = SelfAttention::new(3, 3);
    // sa.set_block(arr2(&[[1.040464,-0.99978536,-0.34397972], [-2.2353952,-1.1311471,-1.0246618], [2.797164,-2.737051,-0.5218795]]));
    // sa.forward_propagate();

    // let mut d = Dense::new(arr1(&[10,8,5]),false);
    // d.set_block(arr1(&[1.040464,-0.99978536,-0.34397972, 1.040464,-0.99978536,-0.34397972, 1.040464,-0.99978536,-0.34397972, 1.040464]));
    // d.forward_propagate();

    // let mut ma = MultiHeadedAttention::new(2, 3, 3);
    // ma.set_block(arr2(&[[1.040464,-0.99978536,-0.34397972], [-2.2353952,-1.1311471,-1.0246618], [2.797164,-2.737051,-0.5218795]]));
    // ma.forward_propagate();

    // let mut an = AddAndNorm::new(3, 3);
    // an.set_block((arr2(&[[0.32770884,-0.51368296,-0.42133167], [-0.28392246,1.2146206,-0.039344255], [0.8881444,0.14447233,-0.8124698]]),arr2(&[[1.040464,-0.99978536,-0.34397972], [-2.2353952,-1.1311471,-1.0246618], [2.797164,-2.737051,-0.5218795]])));
    // an.forward_propagate();

    // let mut e = EncoderBlock::new(3, 3, 3, arr1(&[9,18,9]));
    // e.set_block(arr2(&[[0.32770884,-0.51368296,-0.42133167], [-0.28392246,1.2146206,-0.039344255], [0.8881444,0.14447233,-0.8124698]]));
    // e.forward_propagate();

    // let mut p = PositionalEncoder::new(3, 4);
    // p.set_block(arr2(&[[0.0,0.0,0.0,0.0], [0.0,0.0,0.0,0.0], [0.0,0.0,0.0,0.0]]));
    // p.forward_propagate();

    let mut e = Transformer::new(3, 3, 3, arr1(&[9,400,9]), word_embeddings);
    e.forward_propagate(arr1(&["movie".to_string(), "good".to_string(), "bad".to_string()]));
}