//! The differentiation tape.
//!
//! Nodes carry a globally unique, monotonically increasing id. Because a node is
//! always created after every node it depends on, **descending id order is a valid
//! reverse topological order** — no separate sort pass is needed at backward time.
//!
//! Graphs are reference counted and shared by every [`Var`](crate::autograd::Var)
//! derived from the same root. If two independently rooted graphs ever meet in one
//! expression they are merged, so a leaf created in isolation (a weight penalty,
//! say) can still be combined with a model output.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};

use cubecl::prelude::Runtime;

use crate::backend::FloatElem;
use crate::error::Result;
use crate::tensor::Tensor;

/// Identifier of a node on the tape.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeId(u64);

static NEXT_NODE: AtomicU64 = AtomicU64::new(1);

impl NodeId {
    fn fresh() -> Self {
        NodeId(NEXT_NODE.fetch_add(1, Ordering::Relaxed))
    }

    /// The raw counter value; useful for debugging and for keying gradients.
    pub fn raw(&self) -> u64 {
        self.0
    }
}

/// Identifier of a learnable parameter, stable for the lifetime of the process.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ParamId(u64);

static NEXT_PARAM: AtomicU64 = AtomicU64::new(1);

impl ParamId {
    /// Allocate a fresh parameter id.
    pub fn fresh() -> Self {
        ParamId(NEXT_PARAM.fetch_add(1, Ordering::Relaxed))
    }

    /// The raw counter value.
    pub fn raw(&self) -> u64 {
        self.0
    }
}

/// Given the gradient flowing into a node's output, produce the gradient for each
/// parent, in the order the parents were registered. `None` means "this parent does
/// not receive a gradient from this edge".
pub(crate) type BackwardRule<R, E> =
    Box<dyn Fn(&Tensor<R, E>) -> Result<Vec<Option<Tensor<R, E>>>>>;

pub(crate) struct Node<R: Runtime, E: FloatElem> {
    pub(crate) parents: Vec<NodeId>,
    pub(crate) rule: BackwardRule<R, E>,
    /// Set on leaves that stand for a learnable parameter.
    pub(crate) param: Option<ParamId>,
}

/// A shared, reference-counted tape.
pub struct Graph<R: Runtime, E: FloatElem> {
    nodes: RefCell<HashMap<NodeId, Node<R, E>>>,
}

impl<R: Runtime, E: FloatElem> Default for Graph<R, E> {
    fn default() -> Self {
        Self {
            nodes: RefCell::new(HashMap::new()),
        }
    }
}

impl<R: Runtime, E: FloatElem> Graph<R, E> {
    /// A fresh, empty tape.
    pub fn new() -> Rc<Self> {
        Rc::new(Self::default())
    }

    pub(crate) fn push(&self, parents: Vec<NodeId>, rule: BackwardRule<R, E>) -> NodeId {
        let id = NodeId::fresh();
        self.nodes.borrow_mut().insert(
            id,
            Node {
                parents,
                rule,
                param: None,
            },
        );
        id
    }

    pub(crate) fn push_leaf(&self, param: Option<ParamId>) -> NodeId {
        let id = NodeId::fresh();
        self.nodes.borrow_mut().insert(
            id,
            Node {
                parents: Vec::new(),
                rule: Box::new(|_| Ok(Vec::new())),
                param,
            },
        );
        id
    }

    /// Number of live nodes.
    pub fn len(&self) -> usize {
        self.nodes.borrow().len()
    }

    /// Whether the tape holds no nodes.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Drop every node, releasing all tensors saved for backward.
    pub fn clear(&self) {
        self.nodes.borrow_mut().clear();
    }

    /// Move all of `other`'s nodes into `self`. Ids are globally unique, so this
    /// cannot collide.
    pub(crate) fn absorb(&self, other: &Graph<R, E>) {
        if core::ptr::eq(self, other) {
            return;
        }
        let mut mine = self.nodes.borrow_mut();
        for (id, node) in other.nodes.borrow_mut().drain() {
            mine.insert(id, node);
        }
    }

    pub(crate) fn with_nodes<T>(&self, f: impl FnOnce(&HashMap<NodeId, Node<R, E>>) -> T) -> T {
        f(&self.nodes.borrow())
    }
}

/// Gradients produced by a backward pass.
///
/// Parameter gradients are keyed by [`ParamId`]; gradients for non-parameter nodes
/// (model inputs, intermediate activations) are keyed by [`NodeId`] and are only
/// retained when the backward pass was asked to keep them.
pub struct Grads<R: Runtime, E: FloatElem> {
    params: HashMap<ParamId, Tensor<R, E>>,
    nodes: HashMap<NodeId, Tensor<R, E>>,
}

impl<R: Runtime, E: FloatElem> Default for Grads<R, E> {
    fn default() -> Self {
        Self {
            params: HashMap::new(),
            nodes: HashMap::new(),
        }
    }
}

impl<R: Runtime, E: FloatElem> Grads<R, E> {
    /// The gradient of a parameter, if it participated in the traversed subgraph.
    pub fn get(&self, id: ParamId) -> Option<&Tensor<R, E>> {
        self.params.get(&id)
    }

    /// Take ownership of a parameter's gradient.
    pub fn take(&mut self, id: ParamId) -> Option<Tensor<R, E>> {
        self.params.remove(&id)
    }

    /// The gradient recorded for a node, when node gradients were retained.
    pub fn node(&self, id: NodeId) -> Option<&Tensor<R, E>> {
        self.nodes.get(&id)
    }

    /// Every parameter gradient.
    pub fn iter(&self) -> impl Iterator<Item = (&ParamId, &Tensor<R, E>)> {
        self.params.iter()
    }

    /// Number of parameters that received a gradient.
    pub fn len(&self) -> usize {
        self.params.len()
    }

    /// Whether no gradient was produced.
    pub fn is_empty(&self) -> bool {
        self.params.is_empty()
    }

    /// Insert or accumulate a parameter gradient.
    pub fn accumulate(&mut self, id: ParamId, grad: Tensor<R, E>) -> Result<()> {
        match self.params.remove(&id) {
            Some(existing) => {
                let sum = crate::tensor::ops::elemwise::add(&existing, &grad)?;
                self.params.insert(id, sum);
            }
            None => {
                self.params.insert(id, grad);
            }
        }
        Ok(())
    }

    pub(crate) fn set_node(&mut self, id: NodeId, grad: Tensor<R, E>) {
        self.nodes.insert(id, grad);
    }

    /// Merge another set of gradients into this one, summing where they overlap.
    /// Used for gradient accumulation across micro-batches.
    pub fn merge(&mut self, other: Grads<R, E>) -> Result<()> {
        for (id, grad) in other.params {
            self.accumulate(id, grad)?;
        }
        Ok(())
    }

    /// Scale every gradient in place, e.g. to average across accumulation steps.
    pub fn scale(&mut self, factor: f32) {
        let scaled: HashMap<_, _> = self
            .params
            .drain()
            .map(|(id, g)| (id, crate::tensor::ops::elemwise::mul_scalar(&g, factor)))
            .collect();
        self.params = scaled;
    }
}
