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
fn softmax(mut x: ArrayViewMut1<f32>) {
    // Iterate through the elements of the array to find the highest value.
    let mut highest = 0.0;
    for i in 0..x.len() {
        if x[i] > highest {
            highest = x[i];
        }
    }
    x.mapv_inplace(|e| e - highest); // Subtract the highest value from each element in the array.
    x.mapv_inplace(f32::exp); // Apply the exponential function to each element in the array.

    let norm = x.sum(); // Compute the sum of all elements in the array.

    x.mapv_inplace(|e| e / norm); // Divide each element by the sum to normalize the array.
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
            softmax(x);
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
        // Calculate the error with respect to the input values
        let mut value_error = Array2::<f32>::zeros((self.input.shape()[0], self.input.shape()[1]));
        
        // Iterate over the columns (j) of the input
        for j in 0..self.input.shape()[1] {
            // Iterate over the rows (k) of the input
            for k in 0..self.input.shape()[0] {
                // Iterate over the rows (i) of the input
                for i in 0..self.input.shape()[0] {
                    // Accumulate the error by multiplying the error of the current row (i) with the corresponding weight
                    value_error[[k, j]] += error[[i, j]] * self.weights[[i, k]];
                }
                
                // Iterate over the columns (l) of the input
                for l in 0..self.input.shape()[1] {
                    // Update the parameters using the accumulated error, input value, and learning rate
                    self.params.value[[l, j]] -= value_error[[k, j]] * self.input.index_axis(Axis(0), k)[l] * LR;
                }
            }
        }

        // Calculate the weight update rate
        let mut weight_rate = Array2::<f32>::zeros((self.input.shape()[0], self.input.shape()[1]));
        
        // Iterate over the rows (i) of the input
        for i in 0..self.input.shape()[0] {
            // Iterate over the rows (j) of the input
            for j in 0..self.input.shape()[0] {
                // Iterate over the columns (k) of the input
                for k in 0..self.input.shape()[1] {
                    // Accumulate the weight update rate by multiplying the error of the current row (i)
                    // with the corresponding value vector element
                    weight_rate[[i, j]] += error[[i, k]] * self.value_vecs[[j, k]];
                }
            }
        }

        let mut prev_error = Array2::<f32>::zeros((self.input.shape()[0], self.input.shape()[1]));
        let mut unnormalised_error = Array2::<f32>::zeros((self.input.shape()[0], self.input.shape()[1]));
        let unchanged_key = self.params.key.clone();
        let unchanged_query = self.params.query.clone();

        // Iterate over the rows (i) of the input
        for i in 0..self.input.shape()[0] {
            // Iterate over the rows (j) of the input
            for j in 0..self.input.shape()[0] {
                // Iterate over the rows (k) of the input
                for k in 0..self.input.shape()[0] {
                    // Check if the current row (j) is equal to the current row (k)
                    if j == k {
                        // Calculate the rate of change of the output with respect to the input
                        let output_rate = self.weights[[i, j]] * (1.0 - self.weights[[i, j]]);
                        unnormalised_error[[i, j]] += output_rate * weight_rate[[i, j]];
                    } else {
                        let output_rate = -self.weights[[i, j]] * self.weights[[i, k]];
                        unnormalised_error[[i, j]] += output_rate * weight_rate[[i, j]];
                    }
                }

                // Iterate over the columns (k) of the input
                for k in 0..self.input.shape()[1] {
                    // Calculate the rate of change of the key and query vectors
                    let key_rate = self.vec_query_matrix[[i, j, k]] * unnormalised_error[[i, j]];
                    let query_rate = self.vec_key_matrix[[i, j, k]] * unnormalised_error[[i, j]];

                    // Iterate over the columns (l) of the input
                    for l in 0..self.input.shape()[1] {
                        // Update the key and query parameters using the calculated rates, input values, and learning rate
                        self.params.key[[l, k]] -= key_rate * self.input[[i, l]] * LR;
                        self.params.query[[l, k]] -= query_rate * self.input[[i, l]] * LR;

                        // Accumulate the previous error by multiplying the rates with the unchanged key and query values
                        prev_error[[i, l]] += key_rate * unchanged_key[[l, k]];
                        prev_error[[i, l]] += query_rate * unchanged_query[[l, k]];
                    }
                }
            }
        }

        prev_error
    }
}