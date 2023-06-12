use ndarray::{Array1, Array2};
use crate::add_and_norm::AddAndNorm;
use crate::block::Block;
use crate::multi_headed_attention::MultiHeadedAttention;
use crate::dense::Dense;
use log::info;

// Defines multi headed attention and feed forward blocks.
pub struct EncoderBlockParams {
    multi_headed: MultiHeadedAttention,
    feed_forward: Dense,
}

// Defines encoder block struct
pub struct EncoderBlock {
    input: Array2::<f32>,
    add_and_norm: AddAndNorm,
    rows: usize,
    cols: usize,
    params: EncoderBlockParams,
}

impl EncoderBlock {
    /// Create a new encoder block with the given parameters
    pub fn new(rows: usize, cols: usize, num_heads: usize, layer_sizes: Array1<usize>) -> EncoderBlock {
        let multi_headed = MultiHeadedAttention::new(num_heads, rows, cols);
        let add_and_norm = AddAndNorm::new(rows, cols);
        let feed_forward = Dense::new(layer_sizes, false);

        let params = EncoderBlockParams { multi_headed, feed_forward };

        let block: EncoderBlock = EncoderBlock {
            input: Array2::<f32>::zeros((rows, cols)),
            rows,
            cols,
            add_and_norm,
            params
        };

        block
    }
}

impl Block for EncoderBlock {
    type Input = Array2<f32>;
    type Output = Array2<f32>;

    fn forward_propagate(&mut self, value: Self::Input) -> Self::Output {
        self.input = value;
        info!("Encoder block input: \n {:?}", self.input);

        let multi_out = self.params.multi_headed.forward_propagate(self.input.clone());

        let add_out = self.add_and_norm.forward_propagate((self.input.clone(), multi_out));
        let mut add_out_flat = Array1::<f32>::zeros(self.rows*self.cols);
        for j in 0..self.rows {
            for k in 0..self.cols {
                add_out_flat[j*self.cols + k] = add_out[[j,k]];
            }
        }

        let feed_out = self.params.feed_forward.forward_propagate(add_out_flat);
        let feed_out_sq = feed_out.into_shape([self.rows, self.cols]).unwrap();

        let output = self.add_and_norm.forward_propagate((add_out, feed_out_sq));

        info!("Encoder block output: \n {:?}", output);

        output
    }

    fn back_propagate(&mut self, error: Self::Output) -> Self::Input {
        error
    }
}