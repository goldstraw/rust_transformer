use ndarray::{Array1, Array2};
use crate::block::Block;
use crate::LR;
use rand_distr::{Distribution, Normal};
use log::info;

// Defines struct for storing dense parameters
pub struct DenseParams {
    weights: Vec<Array2::<f32>>,
    biases: Vec<Array1::<f32>>,
}

// Defines dense layer struct
pub struct Dense {
    input: Array1::<f32>,
    pub input_size: usize,
    linear: bool,
    classifier: bool,
    layer: Vec<Array1::<f32>>,
    error: Vec<Array1::<f32>>,
    params: DenseParams,
}

impl Dense {
    /// Create a new self-attention block with the given parameters
    pub fn new(layer_sizes: Array1<usize>, linear: bool, classifier: bool) -> Dense {
        let input = Array1::<f32>::zeros(layer_sizes[0]);
        let mut layer = vec![];
        let mut error = vec![];
        let mut weights = vec![];
        let mut biases = vec![Array1::<f32>::zeros(0)];

        for i in 0..layer_sizes.len()-1 {
            let normal = Normal::new(0.0, (2.0 / layer_sizes[i] as f32).sqrt()).unwrap();
            let mut layer_weights = Array2::<f32>::zeros((layer_sizes[i],layer_sizes[i+1]));
            let mut layer_biases = Array1::<f32>::zeros(layer_sizes[i+1]);

            // Use He initialisation by using a mean of 0.0 and a standard deviation of sqrt(2/n)
            layer_weights.mapv_inplace(|_| normal.sample(&mut rand::thread_rng()));
            layer_biases.mapv_inplace(|_| normal.sample(&mut rand::thread_rng()));

            weights.push(layer_weights);
            biases.push(layer_biases);
            layer.push(Array1::<f32>::zeros(layer_sizes[i]));
            error.push(Array1::<f32>::zeros(layer_sizes[i]));
        }

        layer.push(Array1::<f32>::zeros(layer_sizes[layer_sizes.len()-1]));
        error.push(Array1::<f32>::zeros(layer_sizes[layer_sizes.len()-1]));

        let params = DenseParams { weights, biases };

        let block: Dense = Dense {
            input,
            input_size: layer_sizes[0],
            linear,
            classifier,
            layer,
            error,
            params
        };

        block
    }
}

/// Sigmoid activation function
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Inverse derivative of the sigmoid function
pub fn inv_deriv_sigmoid(x: f32) -> f32 {
    let z: f32 = (x / (1.0 - x)).ln();
    sigmoid(z) * (1.0 - sigmoid(z))
}

impl Block for Dense {
    type Input = Array1<f32>;
    type Output = Array1<f32>;

    fn forward_propagate(&mut self, value: Self::Input) -> Self::Output {
        self.input = value;
        info!("Dense block input: \n {:?}", self.input);

        self.layer[0].assign(&self.input);
        for i in 1..self.layer.len() {
            let weighted_sum = &self.layer[i - 1].dot(&self.params.weights[i - 1]);
            self.layer[i] = weighted_sum + &self.params.biases[i];
            if !self.linear {
                if self.classifier {
                    self.layer[i].mapv_inplace(sigmoid);
                } else {
                    self.layer[i].mapv_inplace(|x| if x > 0.0 { x } else { 0.0 });
                }
            }
        }

        info!("Dense block output: \n {:?}", self.layer[self.layer.len()-1]);

        self.layer[self.layer.len()-1].clone()
    }

    fn back_propagate(&mut self, error: Self::Output) -> Self::Input {
        self.error[self.layer.len()-1] = error;
        for i in 0..self.layer.len()-1 {
            let index: usize = self.layer.len() - (i+2);
            for j in 0..self.layer[index].len() {
                self.error[index][j] = 0.0;
                for k in 0..self.layer[index+1].len() {
                    let next_error: f32 = self.error[index+1][k];
                    self.error[index][j] += self.params.weights[index][[j,k]] * next_error;
                    self.params.weights[index][[j,k]] -= self.layer[index][j] * next_error * LR;
                }
                // The first layer did not have an activation function nor biases
                if index > 0 {
                    // Apply the derivative of the relevant activation function
                    if self.classifier {
                        self.error[index][j] *= inv_deriv_sigmoid(self.layer[index][j]);
                    } else if self.layer[index][j] <= 0.0 {
                        self.error[index][j] = 0.0;
                    }
                    self.params.biases[index][j] -= self.error[index][j] * LR;
                }
            }
        }

        self.error[0].clone()
    }
}