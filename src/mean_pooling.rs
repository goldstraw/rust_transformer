use ndarray::{Axis, Array1, Array2};
use crate::block::Block;

// Defines a mean pooling struct
pub struct MeanPooling {
    input: Array2::<f32>,
}

impl MeanPooling {
    /// Create a new mean pooling block with the given parameters
    pub fn new(rows: usize, cols: usize) -> MeanPooling {
        let block: MeanPooling = MeanPooling {
            input: Array2::<f32>::zeros((rows, cols)),
        };
        block
    }
}

impl Block for MeanPooling {
    type Input = Array2<f32>;
    type Output = Array1<f32>;

    fn forward_propagate(&mut self, value: Self::Input) -> Self::Output {
        self.input = value;
        let output = self.input.mean_axis(Axis(0)).unwrap();

        output
    }

    fn back_propagate(&mut self, error: Self::Output) -> Self::Input {
        let shape = self.input.shape();
        Array2::from_shape_fn([shape[0], shape[1]], |(_,i)| error[i] / shape[0] as f32)
    }
}