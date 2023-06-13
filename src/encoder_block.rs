use ndarray::{Array1, Array2};
use crate::add_and_norm::AddAndNorm;
use crate::block::Block;
use crate::multi_headed_attention::MultiHeadedAttention;
use crate::dense::Dense;

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
        let feed_forward = Dense::new(layer_sizes, false, false);

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

        let multi_out = self.params.multi_headed.forward_propagate(self.input.clone());

        let add_out = self.add_and_norm.forward_propagate((self.input.clone(), multi_out));
        let add_out_flat = add_out.clone().into_shape(self.rows*self.cols).unwrap();

        let feed_out = self.params.feed_forward.forward_propagate(add_out_flat);
        let feed_out_sq = feed_out.into_shape([self.rows, self.cols]).unwrap();

        let output = self.add_and_norm.forward_propagate((add_out, feed_out_sq));

        output
    }

    fn back_propagate(&mut self, error: Self::Output) -> Self::Input {
        let norm_error = self.add_and_norm.back_propagate(error);
        let flat_error = norm_error.1.into_shape(self.rows*self.cols).unwrap();
        let feed_flat_error = self.params.feed_forward.back_propagate(flat_error);
        let feed_error = feed_flat_error.into_shape([self.rows,self.cols]).unwrap();
        let residual_error = &norm_error.0 + &feed_error;
        let norm_error2 = self.add_and_norm.back_propagate(residual_error);
        let multi_headed_error = self.params.multi_headed.back_propagate(norm_error2.1);
        let prev_error = &norm_error2.0 + &multi_headed_error;

        prev_error
    }
}