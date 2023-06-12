use ndarray::Array2;
use crate::block::Block;
use log::info;

// Defines an add and norm struct
pub struct PositionalEncoder {
    input: Array2::<f32>,
    dimensionality: usize,
}

impl PositionalEncoder {
    /// Create a new add and norm block with the given parameters
    pub fn new(rows: usize, cols: usize) -> PositionalEncoder {

        let block: PositionalEncoder = PositionalEncoder {
            input: Array2::<f32>::zeros((rows, cols)),
            dimensionality: cols,
        };

        block
    }
}

impl Block for PositionalEncoder {
    type Input = Array2<f32>;
    type Output = Array2<f32>;

    fn forward_propagate(&mut self, value: Self::Input) -> Self::Output {
        self.input = value;
        info!("Positional encoder block input: \n {:?}", self.input);

        let mut positional_encodings = Array2::<f32>::zeros((self.input.shape()[0], self.dimensionality));
        for i in 0..self.input.shape()[0] {
            for j in 0..self.dimensionality {
                let pos = i as f32;
                let dim_model = self.dimensionality as f32;
                let dim = j as f32;

                let angle = pos / f32::powf(10000.0, 2.0 * dim / dim_model);

                if j % 2 == 0 {
                    positional_encodings[[i,j]] = angle.sin();
                } else {
                    positional_encodings[[i,j]] = angle.cos();
                }
            }
        }

        let output = &positional_encodings + &self.input;

        info!("Positional encoder block output: \n {:?}", output);

        output
    }
}