//! Learnable parameters.
//!
//! A `Param` is a **shared, interior-mutable handle** to a tensor. Cloning one
//! aliases the same storage, which is what lets the optimizer update weights
//! through a flat `Vec<(String, Param)>` without any `&mut` traversal of the model
//! tree — and what lets LoRA freeze a base weight by flipping one flag.

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use cubecl::prelude::Runtime;

use crate::autograd::{ParamId, Var};
use crate::backend::FloatElem;
use crate::tensor::{Shape, Tensor};

/// A learnable tensor.
pub struct Param<R: Runtime, E: FloatElem> {
    id: ParamId,
    value: Rc<RefCell<Tensor<R, E>>>,
    requires_grad: Rc<Cell<bool>>,
}

impl<R: Runtime, E: FloatElem> Clone for Param<R, E> {
    /// Aliases the same storage and the same id.
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            value: self.value.clone(),
            requires_grad: self.requires_grad.clone(),
        }
    }
}

impl<R: Runtime, E: FloatElem> core::fmt::Debug for Param<R, E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "Param#{}({}, requires_grad={})",
            self.id.raw(),
            self.shape(),
            self.requires_grad()
        )
    }
}

impl<R: Runtime, E: FloatElem> Param<R, E> {
    /// Wrap a tensor as a trainable parameter with a fresh id.
    pub fn new(value: Tensor<R, E>) -> Self {
        Self {
            id: ParamId::fresh(),
            value: Rc::new(RefCell::new(value)),
            requires_grad: Rc::new(Cell::new(true)),
        }
    }

    /// This parameter's stable id, used as the gradient and optimizer-state key.
    pub fn id(&self) -> ParamId {
        self.id
    }

    /// The current value (a cheap alias of the device buffer).
    pub fn value(&self) -> Tensor<R, E> {
        self.value.borrow().clone()
    }

    /// Replace the value. Used by optimizers.
    pub fn set(&self, value: Tensor<R, E>) {
        *self.value.borrow_mut() = value;
    }

    /// Shape of the value.
    pub fn shape(&self) -> Shape {
        self.value.borrow().shape().clone()
    }

    /// Number of scalars.
    pub fn numel(&self) -> usize {
        self.value.borrow().len()
    }

    /// Whether gradients should be computed for this parameter.
    pub fn requires_grad(&self) -> bool {
        self.requires_grad.get()
    }

    /// Freeze or unfreeze. A frozen parameter enters the graph as a constant, so
    /// no gradient is computed and no activation is saved for it.
    pub fn set_requires_grad(&self, flag: bool) {
        self.requires_grad.set(flag);
    }

    /// Freeze this parameter.
    pub fn freeze(&self) {
        self.set_requires_grad(false);
    }

    /// A [`Var`] for this parameter, joined to `anchor`'s tape.
    ///
    /// If `anchor` is untracked (inference) or this parameter is frozen, the result
    /// is a constant and costs nothing.
    pub fn var(&self, anchor: &Var<R, E>) -> Var<R, E> {
        if !self.requires_grad() || !crate::autograd::grad_mode::is_enabled() {
            return Var::constant(self.value());
        }
        match anchor.graph() {
            Some(graph) => Var::param_leaf(self.value(), graph, self.id),
            // The anchor carries no tape — every layer before this one is frozen.
            // Start a tape here; `Var::record` merges tapes when they meet.
            None => self.var_standalone(),
        }
    }

    /// A [`Var`] on a brand new tape. Use when a parameter is the root of its own
    /// expression, such as a weight-decay penalty computed outside the model.
    pub fn var_standalone(&self) -> Var<R, E> {
        if self.requires_grad() && crate::autograd::grad_mode::is_enabled() {
            let graph = crate::autograd::Graph::new();
            Var::param_leaf(self.value(), graph, self.id)
        } else {
            Var::constant(self.value())
        }
    }
}
