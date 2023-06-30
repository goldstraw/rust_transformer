use rusttransformer::*;
use log::LevelFilter;
use std::io;

fn main() {
    // Set the custom logger as the global logger
    log::set_logger(&logger::CustomLogger).unwrap();
    log::set_max_level(LevelFilter::Info);

    println!("Enter the max number of words: ");
    let mut input = String::new();
    io::stdin().read_line(&mut input).expect("Failed to read input.");
    let num_words = input.trim().parse().expect("Invalid input.");

    println!("Enter the dimensionality: ");
    input.clear();
    io::stdin().read_line(&mut input).expect("Failed to read input.");
    let dimensionality = input.trim().parse().expect("Invalid input.");

    println!("Enter the number of encoders: ");
    input.clear();
    io::stdin().read_line(&mut input).expect("Failed to read input.");
    let num_encoders = input.trim().parse().expect("Invalid input.");

    println!("Enter the number of heads: ");
    input.clear();
    io::stdin().read_line(&mut input).expect("Failed to read input.");
    let num_heads = input.trim().parse().expect("Invalid input.");

    println!("Enter the hidden layer size: ");
    input.clear();
    io::stdin().read_line(&mut input).expect("Failed to read input.");
    let hidden_layer_size = input.trim().parse().expect("Invalid input.");

    run::run(num_words, dimensionality, num_encoders, num_heads, hidden_layer_size);
}