use ndarray::{arr1, Array1, Array2};
use crate::block::Block;
use crate::self_attention::SelfAttention;
use crate::dense::Dense;
use log::info;

// Defines attention heads and dense layer.
pub struct MultiHeadedAttentionParams {
    heads: Array1::<SelfAttention>,
    linear: Dense,
}

// Defines multi-headed attention struct
pub struct MultiHeadedAttention {
    input: Array2::<f32>,
    params: MultiHeadedAttentionParams,
}

impl MultiHeadedAttention {
    /// Create a new self-attention block with the given parameters
    pub fn new(num_heads: usize, rows: usize, cols: usize) -> MultiHeadedAttention {
        let heads: Array1<SelfAttention> = Array1::from_shape_fn(num_heads, |_| SelfAttention::new(rows, cols));
        let linear: Dense = Dense::new(arr1(&[rows*cols*num_heads, rows*cols]), true);

        let params = MultiHeadedAttentionParams { heads, linear };

        let block: MultiHeadedAttention = MultiHeadedAttention {
            input: Array2::<f32>::zeros((rows, cols)),
            params
        };

        block
    }
}

impl Block for MultiHeadedAttention {
    type Input = Array2<f32>;
    type Output = Array2<f32>;

    fn set_block(&mut self, value: Self::Input) {
        self.input = value;
    }

    fn forward_propagate(&mut self) -> Self::Output {
        info!("Multi-headed attention layer input: \n {:?}", self.input);

        let mut concat_heads = Array1::<f32>::zeros(self.params.linear.input_size);
        let mut concat_index = 0;
        let mut rows = 0;
        let mut cols = 0;
        for i in 0..self.params.heads.len() {
            self.params.heads[i].set_block(self.input.clone());
            let head = self.params.heads[i].forward_propagate();
            rows = head.shape()[0];
            cols = head.shape()[1];
            for j in 0..rows {
                for k in 0..cols {
                    concat_heads[concat_index] = head[[j,k]];
                    concat_index += 1;
                }
            }
        }

        self.params.linear.set_block(concat_heads);
        let output = self.params.linear.forward_propagate();

        info!("Multi-headed attention layer output: \n {:?}", output);

        output.into_shape([rows, cols]).unwrap()
    }
}