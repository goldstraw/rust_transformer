use ndarray::{Array2, Array3, Axis, ArrayViewMut1};
use crate::block::Block;
use crate::LR;
use rand_distr::{Distribution, Normal};

// Defines struct for storing key, query, and value matrices
pub struct SelfAttentionParams {
    key: Array2::<f32>,
    query: Array2::<f32>,
    value: Array2::<f32>,
}

// Defines self-attention struct
pub struct SelfAttention {
    input: Array2::<f32>,
    weights: Array2::<f32>,
    value_vecs: Array2::<f32>,
    vec_key_matrix: Array3::<f32>,
    vec_query_matrix: Array3::<f32>,
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

        // Store intermediary calculations for use in back-propagation
        let weights = Array2::<f32>::zeros((input.shape()[0], input.shape()[1]));
        let value_vecs = Array2::<f32>::zeros((input.shape()[0], input.shape()[1]));
        let vec_key_matrix = Array3::<f32>::zeros((input.shape()[0], input.shape()[0], input.shape()[1]));
        let vec_query_matrix = Array3::<f32>::zeros((input.shape()[0], input.shape()[0], input.shape()[1]));

        let params = SelfAttentionParams { key, query, value };

        let block: SelfAttention = SelfAttention {
            input,
            weights,
            value_vecs,
            vec_key_matrix,
            vec_query_matrix,
            params
        };

        block
    }
}

// Apply softmax normalisation to an Array1.
fn normalise(mut x: ArrayViewMut1<f32>) {
    let mut highest = 0.0;
    for i in 0..x.len() {
        if x[i] > highest {
            highest = x[i];
        }
    }
    x.mapv_inplace(|e| e - highest);
    x.mapv_inplace(f32::exp);
    let norm = x.sum();
    x.mapv_inplace(|e| e / norm);
}

impl Block for SelfAttention {
    type Input = Array2<f32>;
    type Output = Array2<f32>;

    fn forward_propagate(&mut self, value: Self::Input) -> Self::Output {
        self.input = value;

        // Generate context by finding weight vectors
        self.weights = Array2::<f32>::zeros((self.input.shape()[0], self.input.shape()[1]));

        for i in 0..self.input.shape()[0] {
            for j in 0..self.input.shape()[0] {
                // Find similarity of word i and word j by using their dot product
                let vec_i = self.input.index_axis(Axis(0), i);
                // Multiply vector inputs by query and key matrices
                let vec_query = vec_i.dot(&self.params.query);
                let vec_j = &self.input.index_axis(Axis(0), j);
                let vec_key = &vec_j.dot(&self.params.key);
                // Store intermediary values for use in back propagation
                for k in 0..self.input.shape()[1] {
                    self.vec_key_matrix[[i,j,k]] = vec_key[k];
                    self.vec_query_matrix[[i,j,k]] = vec_query[k];
                }
                self.weights[[i,j]] = vec_query.dot(vec_key);
            }
        }

        // Normalize each weight vector using softmax
        for x in self.weights.axis_iter_mut(Axis(0)) {
            normalise(x);
        }

        // Generate output by calculating value vectors
        let mut output = Array2::<f32>::zeros((self.input.shape()[0], self.input.shape()[1]));

        // Apply value matrix to each vector before its use
        self.value_vecs = Array2::<f32>::zeros((self.input.shape()[0], self.input.shape()[1]));
        for i in 0..self.input.shape()[0] {
            let vec_i = self.input.index_axis(Axis(0), i);
            // Multiply each vector inputs by value matrix
            let vec_value = vec_i.dot(&self.params.value);
            self.value_vecs.row_mut(i).assign(&vec_value);
        }

        for i in 0..self.input.shape()[0] {
            for j in 0..self.input.shape()[1] {
                for k in 0..self.input.shape()[0] {
                    output[[i,j]] += self.value_vecs[[k,j]] * self.weights[[i,k]];
                }
            }
        }

        output
    }

    fn back_propagate(&mut self, error: Self::Output) -> Self::Input {
        let mut value_error = Array2::<f32>::zeros((self.input.shape()[0], self.input.shape()[1]));
        for j in 0..self.input.shape()[1] {
            for k in 0..self.input.shape()[0] {
                for i in 0..self.input.shape()[0] {
                    value_error[[k,j]] += error[[i,j]] * self.weights[[i,k]];
                }
                for l in 0..self.input.shape()[1] {
                    self.params.value[[l,j]] -= value_error[[k,j]] * self.input.index_axis(Axis(0), k)[l] * LR;
                }
            }
        }

        let mut weight_rate = Array2::<f32>::zeros((self.input.shape()[0], self.input.shape()[1]));
        for i in 0..self.input.shape()[0] {
            for j in 0..self.input.shape()[0] {
                for k in 0..self.input.shape()[1] {
                    weight_rate[[i,j]] += error[[i,k]] * self.value_vecs[[j,k]];
                }
            }
        }

        let mut prev_error = Array2::<f32>::zeros((self.input.shape()[0], self.input.shape()[1]));
        let mut unnormalised_error = Array2::<f32>::zeros((self.input.shape()[0], self.input.shape()[1]));
        let unchanged_key = self.params.key.clone();
        let unchanged_query = self.params.query.clone();
        for i in 0..self.input.shape()[0] {
            for j in 0..self.input.shape()[0] {
                for k in 0..self.input.shape()[0] {
                    if j == k {
                        // Calculate rate of change of output with respect to input
                        let output_rate = self.weights[[i,j]] * (1.0 - self.weights[[i,j]]);
                        unnormalised_error[[i,j]] += output_rate * weight_rate[[i,j]];
                    } else {
                        let output_rate = - self.weights[[i,j]] * self.weights[[i,k]];
                        unnormalised_error[[i,j]] += output_rate * weight_rate[[i,j]];
                    }
                }

                for k in 0..self.input.shape()[1] {
                    let key_rate = self.vec_query_matrix[[i,j,k]] * unnormalised_error[[i,j]];
                    let query_rate = self.vec_key_matrix[[i,j,k]] * unnormalised_error[[i,j]];
                    for l in 0..self.input.shape()[1] {
                        self.params.key[[l,k]] -= key_rate * self.input[[i,l]] * LR;
                        self.params.query[[l,k]] -= query_rate * self.input[[i,l]] * LR;

                        prev_error[[i,l]] += key_rate * unchanged_key[[l,k]];
                        prev_error[[i,l]] += query_rate * unchanged_query[[l,k]];
                    }
                }
            }
        }

        prev_error
    }
}