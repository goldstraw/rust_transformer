use crate::block::Block;
use ndarray::arr1;
use rand::Rng;
use crate::embedding::load_embeddings;
use crate::transformer::Transformer;
use crate::dataset::load_imdb_dataset;

pub fn run() {
    let word_embeddings = load_embeddings("word_embeddings.json");
    let dataset = load_imdb_dataset("imdb_dataset.csv", 40, word_embeddings.clone());
    let mut transformer = Transformer::new(40, 200, 3, 3, arr1(&[8000,1000,8000]), word_embeddings);
    let mut rng = rand::thread_rng();

    let mut prev_100 = arr1(&[0.0; 100]);
    let mut index = 0;

    loop {
        let example = &dataset[rng.gen_range(0..dataset.len())];
        let val = transformer.forward_propagate(example.review.clone());
        prev_100[index] = (val - example.sentiment).powf(2.0);
        index += 1;
        if index == 100 {
            index = 0;
            println!("{:?}", prev_100.sum() / 100.0);
        }
        transformer.back_propagate(example.sentiment);
    }
}