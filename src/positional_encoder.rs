use ndarray::Array2;
use crate::block::Block;

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
        self.input = value;  // Set the input value for the layer.
    
        // Create positional encodings matrix.
        let mut positional_encodings = Array2::<f32>::zeros((self.input.shape()[0], self.dimensionality));

        // Iterate over rows of the input.
        for i in 0..self.input.shape()[0] {
            // Iterate over columns of the input.
            for j in 0..self.dimensionality {
                // Calculate the angle for positional encoding.
                let angle = i as f32 / f32::powf(10000.0, 2.0 * j as f32 / self.dimensionality as f32);

                // Compute sine or cosine based on the column index.
                positional_encodings[[i,j]] = if j % 2 == 0 { angle.sin() } else { angle.cos() };
            }
        }

        // Add positional encodings to the input.
        let output = &positional_encodings + &self.input;
        output  // Return the output.
    }

    fn back_propagate(&mut self, error: Self::Output) -> Self::Input {
        error  // Return the error for backpropagation.
    }
}