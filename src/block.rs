/// A trait for a block in the transformer
pub trait Block {
    type Input;
    type Output;

    /// Forward propagates input through the block
    fn forward_propagate(&mut self, value: Self::Input) -> Self::Output;

    /// Back propagates error through the block
    fn back_propagate(&mut self, error: Self::Output) -> Self::Input;
}