use ndarray::{Array1, Array2, arr1};
use std::collections::HashMap;
use crate::block::Block;
use crate::dense::{Dense, inv_deriv_sigmoid};
use crate::encoder_block::EncoderBlock;
use crate::mean_pooling::MeanPooling;
use crate::positional_encoder::PositionalEncoder;
use log::info;

// Defines attention heads and dense layer.
pub struct TransformerParams {
    encoder_blocks: Array1::<EncoderBlock>,
}

// Defines multi-headed attention struct
pub struct Transformer {
    input: Array1::<String>,
    output: f32,
    rows: usize,
    cols: usize,
    pos_encoder: PositionalEncoder,
    mean_pool: MeanPooling,
    classification: Dense,
    embedding: HashMap<String, Vec<f32>>,
    params: TransformerParams,
}

impl Transformer {
    /// Create a new self-attention block with the given parameters
    pub fn new(rows: usize, cols: usize, num_heads: usize, layer_sizes: Array1<usize>, embedding: HashMap<String, Vec<f32>>) -> Transformer {
        let encoder_blocks = Array1::from_shape_fn(num_heads, |_| EncoderBlock::new(rows, cols, num_heads, layer_sizes.clone()));
        let params = TransformerParams { encoder_blocks };
        let pos_encoder = PositionalEncoder::new(rows, cols);
        let mean_pool = MeanPooling::new(rows, cols);
        let classification = Dense::new(arr1(&[cols, cols/4 + 1, 1]), false, true);
        let block: Transformer = Transformer {
            input: Array1::from_shape_fn(rows, |_| "".to_string()),
            output: 0.0,
            rows,
            cols,
            pos_encoder,
            mean_pool,
            classification,
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
        info!("Transformer block input: \n {:?}", self.input);

        let embedded = Array2::<f32>::from_shape_fn((self.rows, self.cols), |(i, j)| self.embedding[&self.input[i]][j]);
        let mut enc_output = self.pos_encoder.forward_propagate(embedded);

        for i in 0..self.params.encoder_blocks.len() {
            enc_output = self.params.encoder_blocks[i].forward_propagate(enc_output);
        }

        let pooled_output = self.mean_pool.forward_propagate(enc_output);
        self.output = self.classification.forward_propagate(pooled_output)[0];

        info!("Transformer block output: \n {:?}", self.output);

        self.output
    }

    /// Rather than giving an error here, input a desired value.
    fn back_propagate(&mut self, error: Self::Output) -> Self::Input {
        let last_layer_error = 2.0 * (self.output - error) * inv_deriv_sigmoid(self.output);
        let classification_error = self.classification.back_propagate(arr1(&[last_layer_error]));
        let pool_error = self.mean_pool.back_propagate(classification_error);

        let mut encoder_error = pool_error;
        for i in (0..self.params.encoder_blocks.len()).rev() {
            encoder_error = self.params.encoder_blocks[i].back_propagate(encoder_error);
        }

        // The positional encoder doesn't have any trainable parameters
        // self.pos_encoder.back_propagate(encoder_error);

        arr1(&["".to_string()])
    }
}