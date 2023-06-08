use ndarray::{Array1, Array2, Axis, ArrayViewMut1};
use crate::block::Block;

// Defines self-attention struct
pub struct SelfAttention {
    input: Array2<f32>,
}

impl SelfAttention {
    /// Create a new self-attention block with the given parameters
    pub fn new(rows: usize, cols: usize) -> SelfAttention {
        let input = Array2::<f32>::zeros((rows, cols));

        let block: SelfAttention = SelfAttention {
            input,
        };

        block
    }
}

// Apply softmax normalisation to an Array1.
fn normalise(mut x: ArrayViewMut1<f32>) -> ArrayViewMut1<f32> {
    x.mapv_inplace(f32::exp);
    let norm = x.sum();
    x.mapv_inplace(|e| e / norm);
    x
}

impl Block for SelfAttention {
    type Output = Array2<f32>;

    fn forward_propagate(&self) -> Self::Output {
        // Generate context by finding weight vectors
        let mut weights = Array2::<f32>::zeros((self.input.shape()[0], self.input.shape()[1]));

        for i in 0..self.input.len() {
            for j in 0..self.input.len() {
                // Find similarity of word i and word j by using their dot product
                let vec_i = self.input.index_axis(Axis(0), i);
                let vec_j = &self.input.index_axis(Axis(0), j);
                weights[[i,j]] = vec_i.dot(vec_j);
            }
        }

        // Normalize each weight vector using softmax
        weights.map_axis_mut(Axis(0), normalise);

        // Generate output by calculating value vectors
        let mut output = Array2::<f32>::zeros((self.input.shape()[0], self.input.shape()[1]));

        for i in 0..self.input.len() {
            for j in 0..self.input.len() {
                for k in 0..self.input.len() {
                    output[[i,j]] += &self.input[[k,j]] * weights[[i,k]];
                }
            }
        }

        output
    }
}

struct Dot {
    input: (Array1<f32>, Array1<f32>),
}

impl Block for Dot {
    type Output = Array1<f32>;

    fn forward_propagate(&self) -> Self::Output {
        let mut output = Array1::<f32>::zeros(self.input.0.len());
        for i in 0..self.input.0.len() {
            output[i] = self.input.0[i] * self.input.1[i];
        }

        output
    }
}