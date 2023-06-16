use crate::block::Block;
use ndarray::arr1;
use rand::Rng;
use crate::embedding::load_embeddings;
use crate::transformer::Transformer;
use crate::dataset::load_imdb_dataset;
use log::info;
use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use crate::LR;
use std::time::SystemTime;

pub fn run(num_words: usize, dimensionality: usize, num_encoders: usize, num_heads: usize, hidden_layer_size: usize) {
    let prog_start = SystemTime::now();

    let word_embeddings = load_embeddings("word_embeddings.json");
    let dataset = load_imdb_dataset("imdb_dataset.csv", num_words, word_embeddings.clone());
    let mut transformer = Transformer::new(num_words, dimensionality, num_encoders, num_heads, arr1(&[num_words*dimensionality,hidden_layer_size,num_words*dimensionality]), word_embeddings);
    let mut rng = rand::thread_rng();

    const N: usize = 1000;
    let mut prev_100 = arr1(&[0.0; N]);
    let mut index = 0;
    let test_gaps = 5;
    let mut test_count = 0;
    const TEST_SIZE: usize = 200;

    loop {
        let example = &dataset[rng.gen_range(0..dataset.len()-TEST_SIZE)];
        let val = transformer.forward_propagate(example.review.clone());
        prev_100[index] = (val - example.sentiment).powf(2.0);
        index += 1;
        if index == N {
            index = 0;
            test_count += 1;
            info!("{:?}", prev_100.sum() / N as f32);

            let file_path = format!("MW{}_D{}_E{}_H{}_Hi{}_LR{}", num_words, dimensionality, num_encoders, num_heads, hidden_layer_size, LR);
            let current_time = SystemTime::now()
                .duration_since(prog_start)
                .expect("Failed to get current time")
                .as_secs();
            let line_to_add = format!("{},{}", current_time, prev_100.sum() / N as f32);

            // Open the file in append mode or create it if it doesn't exist
            let file = OpenOptions::new()
                .create(true)
                .append(true)
                .open(file_path)
                .expect("Failed to open file");

            let mut file_writer = BufWriter::new(file);
            writeln!(file_writer, "{}", line_to_add).expect("Failed to write to file");

            if test_count == test_gaps {
                test_count = 0;
                let mut test_cost = arr1(&[0.0; TEST_SIZE]);
                for i in 0..TEST_SIZE {
                    let example = &dataset[dataset.len()-TEST_SIZE+i];
                    let val = transformer.forward_propagate(example.review.clone());
                    test_cost[i] = (val - example.sentiment).powf(2.0);
                }
                info!("TEST - {:?}", test_cost.sum() / TEST_SIZE as f32);

                let file_path = format!("TMW{}_D{}_E{}_H{}_Hi{}_LR{}", num_words, dimensionality, num_encoders, num_heads, hidden_layer_size, LR);
                let current_time = SystemTime::now()
                    .duration_since(prog_start)
                    .expect("Failed to get current time")
                    .as_secs();
                let line_to_add = format!("{},{}", current_time, test_cost.sum() / TEST_SIZE as f32);

                // Open the file in append mode or create it if it doesn't exist
                let file = OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(file_path)
                    .expect("Failed to open file");

                let mut file_writer = BufWriter::new(file);
                writeln!(file_writer, "{}", line_to_add).expect("Failed to write to file");
            }
        }
        transformer.back_propagate(example.sentiment);
    }
}