use ndarray::{Array1, Array2, arr1};
use std::collections::HashMap;
use crate::block::Block;
use crate::encoder_block::EncoderBlock;
use crate::positional_encoder::PositionalEncoder;
use log::info;

// Defines attention heads and dense layer.
pub struct EncoderParams {
    encoder_blocks: Array1::<EncoderBlock>,
}

// Defines multi-headed attention struct
pub struct Encoder {
    input: Array1::<String>,
    rows: usize,
    cols: usize,
    pos_encoder: PositionalEncoder,
    embedding: HashMap<String, Vec<f32>>,
    params: EncoderParams,
}

impl Encoder {
    /// Create a new self-attention block with the given parameters
    pub fn new(rows: usize, cols: usize, num_heads: usize, layer_sizes: Array1<usize>, embedding: HashMap<String, Vec<f32>>) -> Encoder {
        let encoder_blocks = Array1::from_shape_fn(num_heads, |_| EncoderBlock::new(rows, cols, num_heads, layer_sizes.clone()));
        let params = EncoderParams { encoder_blocks };
        let pos_encoder = PositionalEncoder::new(rows, cols);
        let block: Encoder = Encoder {
            input: Array1::from_shape_fn(rows, |_| "".to_string()),
            rows,
            cols,
            pos_encoder,
            embedding,
            params
        };

        block
    }
}

impl Block for Encoder {
    type Input = Array1<String>;
    type Output = Array2<f32>;

    fn forward_propagate(&mut self, value: Self::Input) -> Self::Output {
        self.input = value;
        info!("Encoder block input: \n {:?}", self.input);

        let mut embedded = Array2::<f32>::zeros((self.rows, self.cols));
        for i in 0..self.rows {
            let vec = self.embedding[&self.input[i]].clone();
            for j in 0..self.cols {
                embedded[[i,j]] = vec[j];
            }
        }

        let mut output = self.pos_encoder.forward_propagate(embedded);

        for i in 0..self.params.encoder_blocks.len() {
            output = self.params.encoder_blocks[i].forward_propagate(output);
        }

        info!("Encoder block output: \n {:?}", output);

        output
    }

    fn back_propagate(&mut self, error: Self::Output) -> Self::Input {
        arr1(&["a".to_string()])
    }
}