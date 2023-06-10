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

    fn set_block(&mut self, value: Self::Input) {
        self.input = value;
    }

    fn forward_propagate(&mut self) -> Self::Output {
        info!("Encoder block input: \n {:?}", self.input);

        self.params.multi_headed.set_block(self.input.clone());
        let multi_out = self.params.multi_headed.forward_propagate();

        self.add_and_norm.set_block((self.input.clone(), multi_out));
        let add_out = self.add_and_norm.forward_propagate();
        let mut add_out_flat = Array1::<f32>::zeros(self.rows*self.cols);
        for j in 0..self.rows {
            for k in 0..self.cols {
                add_out_flat[j*self.cols + k] = add_out[[j,k]];
            }
        }

        self.params.feed_forward.set_block(add_out_flat);
        let feed_out = self.params.feed_forward.forward_propagate();
        let feed_out_sq = feed_out.into_shape([self.rows, self.cols]).unwrap();

        self.add_and_norm.set_block((add_out, feed_out_sq));
        let output = self.add_and_norm.forward_propagate();

        info!("Encoder block output: \n {:?}", output);

        output
    }
}