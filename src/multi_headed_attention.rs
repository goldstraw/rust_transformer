use ndarray::{arr1, Array1, Array2, Axis};
use crate::block::Block;
use crate::self_attention::SelfAttention;
use crate::dense::Dense;

// Defines attention heads and dense layer.
pub struct MultiHeadedAttentionParams {
    heads: Array1::<SelfAttention>,
    linear: Dense,
}

// Defines multi-headed attention struct
pub struct MultiHeadedAttention {
    input: Array2::<f32>,
    rows: usize,
    cols: usize,
    num_heads: usize,
    params: MultiHeadedAttentionParams,
}

impl MultiHeadedAttention {
    /// Create a new self-attention block with the given parameters
    pub fn new(num_heads: usize, rows: usize, cols: usize) -> MultiHeadedAttention {
        let heads: Array1<SelfAttention> = Array1::from_shape_fn(num_heads, |_| SelfAttention::new(rows, cols));
        let linear: Dense = Dense::new(arr1(&[rows*cols*num_heads, rows*cols]), true, false);

        let params = MultiHeadedAttentionParams { heads, linear };

        let block: MultiHeadedAttention = MultiHeadedAttention {
            input: Array2::<f32>::zeros((rows, cols)),
            rows,
            cols,
            num_heads,
            params
        };

        block
    }
}

impl Block for MultiHeadedAttention {
    type Input = Array2<f32>;
    type Output = Array2<f32>;

    fn forward_propagate(&mut self, value: Self::Input) -> Self::Output {
        self.input = value;

        let mut concat_heads = Array1::<f32>::zeros(self.params.linear.input_size);
        let mut concat_index = 0;
        for i in 0..self.params.heads.len() {
            let head = self.params.heads[i].forward_propagate(self.input.clone());
            for j in 0..self.input.shape()[0] {
                for k in 0..self.input.shape()[1] {
                    concat_heads[concat_index] = head[[j,k]];
                    concat_index += 1;
                }
            }
        }

        let output = self.params.linear.forward_propagate(concat_heads);

        output.into_shape([self.input.shape()[0], self.input.shape()[1]]).unwrap()
    }

    fn back_propagate(&mut self, error: Self::Output) -> Self::Input {
        let flat_error = error.into_shape(self.rows*self.cols).unwrap();
        let linear_error = self.params.linear.back_propagate(flat_error);
        let mut prev_error = Array2::<f32>::zeros((self.rows,self.cols));
        let multi_headed_error = linear_error.into_shape([self.num_heads,self.rows,self.cols]).unwrap();
        for i in 0..self.num_heads {
            let head_error = multi_headed_error.index_axis(Axis(0), i).to_owned();
            let prev_head_error = self.params.heads[i].back_propagate(head_error);
            prev_error = &prev_error + &prev_head_error;
        }

        prev_error
    }
}