use ndarray::{Array1, Array2, Axis, ArrayViewMut1};
use crate::block::Block;
use rand_distr::{Distribution, Normal};
use log::info;

// Defines struct for storing key, query, and value matrices
pub struct SelfAttentionParams {
    key: Array2::<f32>,
    query: Array2::<f32>,
    value: Array2::<f32>,
}

// Defines self-attention struct
pub struct SelfAttention {
    input: Array2::<f32>,
    params: SelfAttentionParams,
}

impl SelfAttention {
    /// Create a new self-attention block with the given parameters
    pub fn new(rows: usize, cols: usize) -> SelfAttention {
        let input = Array2::<f32>::zeros((rows, cols));
        let mut key = Array2::<f32>::zeros((cols, cols));
        let mut query = Array2::<f32>::zeros((cols, cols));
        let mut value = Array2::<f32>::zeros((cols, cols));

        // Use He initialisation by using a mean of 0.0 and a standard deviation of sqrt(2/n)
        let normal = Normal::new(0.0, (2.0/(rows*cols) as f32).sqrt()).unwrap();
        key.mapv_inplace(|_| normal.sample(&mut rand::thread_rng()));
        query.mapv_inplace(|_| normal.sample(&mut rand::thread_rng()));
        value.mapv_inplace(|_| normal.sample(&mut rand::thread_rng()));

        let params = SelfAttentionParams { key, query, value };

        let block: SelfAttention = SelfAttention {
            input,
            params
        };

        block
    }
}

// Apply softmax normalisation to an Array1.
fn normalise(mut x: ArrayViewMut1<f32>) {
    x.mapv_inplace(f32::exp);
    let norm = x.sum();
    x.mapv_inplace(|e| e / norm);
}

impl Block for SelfAttention {
    type Input = Array2<f32>;
    type Output = Array2<f32>;

    fn forward_propagate(&mut self, value: Self::Input) -> Self::Output {
        self.input = value;
        info!("Self-attention block input: \n {:?}", self.input);

        // Generate context by finding weight vectors
        let mut weights = Array2::<f32>::zeros((self.input.shape()[0], self.input.shape()[1]));

        for i in 0..self.input.shape()[0] {
            for j in 0..self.input.shape()[0] {
                // Find similarity of word i and word j by using their dot product
                let vec_i = self.input.index_axis(Axis(0), i);
                // Multiply vector inputs by query and key matrices
                let vec_query = vec_i.dot(&self.params.query);
                let vec_j = &self.input.index_axis(Axis(0), j);
                let vec_key = &vec_j.dot(&self.params.key);
                weights[[i,j]] = vec_query.dot(vec_key);
            }
        }

        // Normalize each weight vector using softmax
        weights.map_axis_mut(Axis(1), normalise);

        // Generate output by calculating value vectors
        let mut output = Array2::<f32>::zeros((self.input.shape()[0], self.input.shape()[1]));

        // Apply value matrix to each vector before its use
        let mut value_vecs = Array2::<f32>::zeros((self.input.shape()[0], self.input.shape()[1]));
        for i in 0..self.input.shape()[0] {
            let vec_i = self.input.index_axis(Axis(0), i);
            // Multiply each vector inputs by value matrix
            let vec_value = vec_i.dot(&self.params.value);
            value_vecs.row_mut(i).assign(&vec_value);
        }

        for i in 0..self.input.shape()[0] {
            for j in 0..self.input.shape()[1] {
                for k in 0..self.input.shape()[0] {
                    output[[i,j]] += value_vecs[[k,j]] * weights[[i,k]];
                }
            }
        }

        info!("Self-attention block output: \n {:?}", output);

        output
    }
}