use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use serde::{Serialize, Deserialize};
use log::info;

// Defines the embedding object
#[derive(Serialize, Deserialize)]
pub struct Embedding {
    data: HashMap<String, Vec<f32>>,
}

pub fn load_embeddings(file_name: &str) -> HashMap<String, Vec<f32>> {
    let mut file = File::open(file_name).expect("Failed to open file");
    let mut serialized = String::new();
    file.read_to_string(&mut serialized).expect("Failed to read file");

    // Deserialize the embeddings into a HashMap.
    let deserialized: Embedding = serde_json::from_str(&serialized).unwrap();
    let embeddings: HashMap<String, Vec<f32>> = deserialized.data.clone();

    info!("Loaded {} word embeddings successfully.", embeddings.len());

    embeddings
}