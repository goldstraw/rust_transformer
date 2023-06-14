use crate::block::Block;
use ndarray::arr1;
use rand::Rng;
use crate::embedding::load_embeddings;
use crate::transformer::Transformer;
use crate::dataset::load_imdb_dataset;
use log::info;

pub fn run() {
    let word_embeddings = load_embeddings("word_embeddings.json");
    let dataset = load_imdb_dataset("imdb_dataset.csv", 40, word_embeddings.clone());
    let mut transformer = Transformer::new(40, 200, 3, 3, arr1(&[8000,1000,8000]), word_embeddings);
    let mut rng = rand::thread_rng();

    const N: usize = 100;
    let mut prev_100 = arr1(&[0.0; N]);
    let mut index = 0;

    loop {
        let example = &dataset[rng.gen_range(0..dataset.len())];
        let val = transformer.forward_propagate(example.review.clone());
        prev_100[index] = (val - example.sentiment).powf(2.0);
        index += 1;
        if index == N {
            index = 0;
            info!("{:?}", prev_100.sum() / N as f32);
        }
        transformer.back_propagate(example.sentiment);
    }
}