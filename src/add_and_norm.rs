use ndarray::{Axis,Array2};
use crate::block::Block;

// Defines an add and norm struct
pub struct AddAndNorm {
    original_input: Array2::<f32>,
    modified_input: Array2::<f32>,
}

impl AddAndNorm {
    /// Create a new add and norm block with the given parameters
    pub fn new(rows: usize, cols: usize) -> AddAndNorm {

        let block: AddAndNorm = AddAndNorm {
            original_input: Array2::<f32>::zeros((rows, cols)),
            modified_input: Array2::<f32>::zeros((rows, cols)),
        };

        block
    }
}

impl Block for AddAndNorm {
    type Input = (Array2<f32>, Array2<f32>);
    type Output = Array2<f32>;

    fn forward_propagate(&mut self, value: Self::Input) -> Self::Output {
        self.original_input = value.0;
        self.modified_input = value.1;

        let mut output = &self.original_input + &self.modified_input;
        for mut x in output.axis_iter_mut(Axis(0)) {
            let sum_sq = x.mapv(|x| x*x).sum();
            let n = x.len();
            let mean = x.sum() / n as f32;
            let mean_sq = sum_sq / n as f32;
            let stdev = (mean_sq - mean.powf(2.0)).powf(0.5);

            x.mapv_inplace(|y| (y - mean) / stdev);
        }

        output
    }

    fn back_propagate(&mut self, error: Self::Output) -> Self::Input {
        // Each input element in the word vector affects the output in multiple
        // ways as it's used in the stdev and mean calcs, so each word vector has
        // a Jacobean for its dC / dx

        let shape = (error.shape()[0], error.shape()[1]);
        let mut prev_error: Self::Input = (Array2::<f32>::zeros(shape), Array2::<f32>::zeros(shape));

        let input = &self.original_input + &self.modified_input;
        for (count, x) in input.axis_iter(Axis(0)).enumerate() {
            let n = x.len() as f32;
            let i = Array2::<f32>::eye(n as usize);
            let mean = x.mean().unwrap();
            let stdev = x.std(0.0);
            // Make a matrix from (xi-μ) * (xj-μ) for use in the jacobean
            let x_matrix = Array2::from_shape_fn((n as usize, n as usize), |(i, j)| (&x - mean)[i] * (&x - mean)[j]);
            let jacobean = ((i * n) - 1.0) / (n * stdev) - (x_matrix / (n * stdev.powi(3)));
            // Calculate all the dC / dx for each input x. There will be n rates of change per input element.
            let p = Array2::from_shape_fn((n as usize, n as usize), |(i, j)| error[[count, i]] * jacobean[[i, j]]);
            // Sum each rate of change for each input to get the final dC / dx.
            let row_error = p.sum_axis(Axis(0));
            for j in 0..error.shape()[1] {
                prev_error.0[[count, j]] = row_error[j];
                prev_error.1[[count, j]] = row_error[j];
            }
        }

        prev_error
    }
}