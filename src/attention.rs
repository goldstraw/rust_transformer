use crate::block::Block;

// Defines single-headed attention struct
pub struct Attention {
    _foobar: usize,
}

impl Attention {
    /// Create a new single-headed attention block with the given parameters
    pub fn new(
        _foobar: usize,
    ) -> Attention {
        let foobar = 0;

        let block: Attention = Attention {
            _foobar: foobar,
        };

        block
    }
}

impl Block for Attention {
    fn forward_propagate(&mut self, input: Vec<Vec<Vec<f32>>>) -> Vec<Vec<Vec<f32>>> {
        input
    }
}