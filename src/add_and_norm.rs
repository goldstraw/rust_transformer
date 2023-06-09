use ndarray::Array2;
use crate::block::Block;
use log::info;

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

    fn set_block(&mut self, value: Self::Input) {
        self.original_input = value.0;
        self.modified_input = value.1;
    }

    fn forward_propagate(&mut self) -> Self::Output {
        info!("Add and Norm block unmodified input: \n {:?}", self.original_input);
        info!("Add and Norm block modified input: \n {:?}", self.modified_input);

        let mut output = &self.original_input + &self.modified_input;
        let sum_sq = output.mapv(|x| x*x).sum();
        let n = output.len();
        let mean = output.sum() / n as f32;
        let mean_sq = sum_sq / n as f32;
        let stdev = (mean_sq - mean.powf(2.0)).powf(0.5);

        output.mapv_inplace(|x| (x - mean) / stdev);

        info!("Add and Norm block output: \n {:?}", output);

        output
    }
}