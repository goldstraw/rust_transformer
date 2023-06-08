/// A trait for a block in the transformer
pub trait Block {
    type Output;

    /// Forward propagates input through the block
    fn forward_propagate(&self) -> Self::Output;
}