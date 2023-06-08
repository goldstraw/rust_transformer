/// A trait for a block in the transformer
pub trait Block {
    type Input;
    type Output;

    /// Sets the input of the block
    fn set_block(&mut self, value: Self::Input);

    /// Forward propagates input through the block
    fn forward_propagate(&self) -> Self::Output;
}