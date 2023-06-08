/// A trait for a block in the transformer
pub trait Block {
    /// Forward propagates input through the block
    fn forward_propagate(&mut self, input: Vec<Vec<Vec<f32>>>) -> Vec<Vec<Vec<f32>>>;
}