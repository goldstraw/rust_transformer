use ndarray::{Array1, Array2, arr1};
use std::collections::HashMap;
use crate::block::Block;
use crate::dense::{Dense, inv_deriv_sigmoid};
use crate::encoder_block::EncoderBlock;
use crate::positional_encoder::PositionalEncoder;

// Defines attention heads and dense layer.
pub struct TransformerParams {
    encoder_blocks: Array1::<EncoderBlock>,
}

// Defines multi-headed attention struct
pub struct Transformer {
    input: Array1::<String>,
    output: f32,
    num_words: usize,
    dimensionality: usize,
    pos_encoder: PositionalEncoder,
    classifier: Dense,
    embedding: HashMap<String, Vec<f32>>,
    params: TransformerParams,
}

impl Transformer {
    /// Create a new self-attention block with the given parameters
    pub fn new(num_words: usize, dimensionality: usize, num_encoders: usize, num_heads: usize, layer_sizes: Array1<usize>, embedding: HashMap<String, Vec<f32>>) -> Transformer {
        let encoder_blocks = Array1::from_shape_fn(num_encoders, |_| EncoderBlock::new(num_words, dimensionality, num_heads, layer_sizes.clone()));
        let params = TransformerParams { encoder_blocks };
        let pos_encoder = PositionalEncoder::new(num_words, dimensionality);
        let classifier = Dense::new(arr1(&[num_words*dimensionality, 1]), false, true);
        let block: Transformer = Transformer {
            input: Array1::from_shape_fn(num_words, |_| "".to_string()),
            output: 0.0,
            num_words,
            dimensionality,
            pos_encoder,
            classifier,
            embedding,
            params
        };

        block
    }
}

impl Block for Transformer {
    type Input = Array1<String>;
    type Output = f32;

    fn forward_propagate(&mut self, value: Self::Input) -> Self::Output {
        self.input = value;
        let embedded = Array2::<f32>::from_shape_fn((self.num_words, self.dimensionality), |(i, j)| self.embedding[&self.input[i]][j]);
        let mut enc_output = self.pos_encoder.forward_propagate(embedded);

        for i in 0..self.params.encoder_blocks.len() {
            enc_output = self.params.encoder_blocks[i].forward_propagate(enc_output);
        }

        let flat_output = enc_output.clone().into_shape(self.num_words*self.dimensionality).unwrap();
        self.output = self.classifier.forward_propagate(flat_output)[0];

        self.output
    }

    /// Rather than giving an error here, input a desired value.
    fn back_propagate(&mut self, error: Self::Output) -> Self::Input {
        let last_layer_error = 2.0 * (self.output - error) * inv_deriv_sigmoid(self.output);
        let classifier_error = self.classifier.back_propagate(arr1(&[last_layer_error]));
        let mut encoder_error = classifier_error.into_shape((self.num_words, self.dimensionality)).unwrap();

        for i in (0..self.params.encoder_blocks.len()).rev() {
            encoder_error = self.params.encoder_blocks[i].back_propagate(encoder_error);
        }

        // The positional encoder doesn't have any trainable parameters
        // self.pos_encoder.back_propagate(encoder_error);

        arr1(&["".to_string()])
    }
}