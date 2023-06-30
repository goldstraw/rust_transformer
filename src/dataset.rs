use ndarray::Array1;
use std::collections::HashMap;

pub struct Review {
    pub review: Array1<String>,
    pub sentiment: f32,
}

/// Clean the review by removing all non-alphanumeric characters
/// and un-encoded words
fn clean_review(mut review: String, word_embeddings: HashMap<String, Vec<f32>>) -> String {
    // "I love this movie! It's so good."
    // => "i love this movie it's so good "
    // Remove "<br" occurrences, as they are likely HTML tags
    review = review.replace("<br", "");

    let mut clean_review: String = String::new();
    let mut in_word = false; 
    let mut current_word = String::new();

    for character in review.chars() {
        if character.is_alphanumeric() || (character == '\'' && in_word) {
            if !in_word {
                in_word = true;
            }
            current_word.push(character);
        } else {
            // If we were inside a word, it has ended, so process it
            if in_word {
                in_word = false;
                if current_word != "" {
                    current_word = current_word.to_lowercase();
                    if word_embeddings.contains_key(&current_word) {
                        current_word.push(' ');
                        clean_review.push_str(&current_word);
                    }
                    current_word = String::new();
                }
            }
        }
    }

    // Process the last word if there is one
    if current_word != "" {
        current_word = current_word.to_lowercase();
        if word_embeddings.contains_key(&current_word) {
            current_word.push(' ');
            clean_review.push_str(&current_word);
        }
    }
    clean_review.to_string()
}

/// Pads the review with empty strings to the desired length
fn pad_review(review: String, review_size: usize) -> Array1<String> {
    let words: Vec<&str> = review.split_whitespace().collect();
    let mut padded_review = Vec::with_capacity(review_size);

    for i in 0..review_size {
        let word = if i < words.len() {
            words[i].to_string()
        } else {
            "".to_string()
        };

        padded_review.push(word);
    }

    Array1::<String>::from_vec(padded_review)
}

pub fn load_imdb_dataset(path: &str, review_size: usize, word_embeddings: HashMap<String, Vec<f32>>) -> Vec<Review> {
    let mut imdb_dataset = Vec::new();
    let mut reader = csv::Reader::from_path(path).unwrap();
    for result in reader.records() {
        let record = result.unwrap();
        let cleaned = clean_review(record[0].to_string(), word_embeddings.clone());
        let review = pad_review(cleaned, review_size);
        let imdb_review = Review {
            review,
            sentiment: if record[1].to_string() == "positive" {1.0} else {0.0},
        };
        imdb_dataset.push(imdb_review);
    }
    imdb_dataset
}