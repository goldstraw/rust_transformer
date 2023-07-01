use crate::block::Block;
use ndarray::arr1;
use rand::Rng;
use crate::embedding::load_embeddings;
use crate::transformer::Transformer;
use crate::dataset::load_imdb_dataset;
use log::info;

pub fn run(num_words: usize, dimensionality: usize, num_encoders: usize, num_heads: usize, hidden_layer_size: usize) {
    let word_embeddings = load_embeddings("word_embeddings.json");
    let dataset = load_imdb_dataset("imdb_dataset.csv", num_words, word_embeddings.clone());
    let mut transformer = Transformer::new(num_words, dimensionality, num_encoders, num_heads, arr1(&[num_words*dimensionality,hidden_layer_size,num_words*dimensionality]), word_embeddings);
    let mut rng = rand::thread_rng();

    const N: usize = 1000; // Number of values to average over
    let mut prev_n = arr1(&[0.0; N]); // Previous N values
    let mut index = 0; // Index of prev_n
    let test_gaps = 5; // Test runs every N * test_gaps iterations
    let mut test_count = 0; 
    const TEST_SIZE: usize = 200; // Number of examples to test on

    loop {
        // Select a random example from the dataset excluding the test set
        let example = &dataset[rng.gen_range(0..dataset.len()-TEST_SIZE)];
        
        // Forward propagate the example through the transformer model
        let val = transformer.forward_propagate(example.review.clone());

        // Calculate the squared difference between the predicted value and the actual sentiment
        prev_n[index] = (val - example.sentiment).powf(2.0);
        index += 1;

        if index == N {
            index = 0;
            test_count += 1;
            // Calculate and log the average loss for the current batch
            info!("{:?}", prev_n.sum() / N as f32);

            // Check if it's time to perform a test on the test set
            if test_count == test_gaps {
                // Reset the test count
                test_count = 0;

                // Create an array to store the test losses
                let mut test_cost = arr1(&[0.0; TEST_SIZE]);

                // Calculate the loss for each example in the test set
                for i in 0..TEST_SIZE {
                    let example = &dataset[dataset.len()-TEST_SIZE+i];
                    let val = transformer.forward_propagate(example.review.clone());
                    test_cost[i] = (val - example.sentiment).powf(2.0);
                }

                // Calculate and log the average loss for the test set
                info!("TEST - {:?}", test_cost.sum() / TEST_SIZE as f32);
            }
        }
    
        // Back propagate the sentiment through the transformer model
        transformer.back_propagate(example.sentiment);
    }
}