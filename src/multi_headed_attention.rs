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

        // Initialize an array to store the concatenated outputs from different heads
        let mut concat_heads = Array1::<f32>::zeros(self.params.linear.input_size);
    
        // Variable to keep track of the current index in the concatenated heads array
        let mut concat_index = 0;
    
        // Iterate through each head in the model's parameters
        for i in 0..self.params.heads.len() {
            // Forward propagate the input through the current head
            let head = self.params.heads[i].forward_propagate(self.input.clone());
    
            // Flatten the head output and concatenate it to the concat_heads array
            for j in 0..self.input.shape()[0] {
                for k in 0..self.input.shape()[1] {
                    concat_heads[concat_index] = head[[j, k]];
                    concat_index += 1;
                }
            }
        }

        // Forward propagate the concatenated heads through the linear layer
        let output = self.params.linear.forward_propagate(concat_heads);

        // Reshape the output to match the shape of the input
        output.into_shape([self.input.shape()[0], self.input.shape()[1]]).unwrap()
    }

    fn back_propagate(&mut self, error: Self::Output) -> Self::Input {
        // Flatten the error tensor into a 1D array
        let flat_error = error.into_shape(self.rows*self.cols).unwrap();

        // Backpropagate the flat error through the linear layer
        let linear_error = self.params.linear.back_propagate(flat_error);

        // Initialize an empty array to store the accumulated error from all heads
        let mut prev_error = Array2::<f32>::zeros((self.rows,self.cols));

        // Reshape the linear error into a multi-headed error tensor
        let multi_headed_error = linear_error.into_shape([self.num_heads,self.rows,self.cols]).unwrap();

        // Iterate over each head and backpropagate the error
        for i in 0..self.num_heads {
            // Extract the error for the current head
            let head_error = multi_headed_error.index_axis(Axis(0), i).to_owned();

            // Backpropagate the head error through the head layer
            let prev_head_error = self.params.heads[i].back_propagate(head_error);

            // Accumulate the previous head error with the overall previous error
            prev_error = &prev_error + &prev_head_error;
        }

        // Return the accumulated previous error
        prev_error
    }
}