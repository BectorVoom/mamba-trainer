//! `Var`: a tensor that may be recorded on a tape.
//!
//! A `Var` with no tape is exactly as cheap as the tensor it wraps — one `Option`
//! check per op — which is why inference and training share a single code path
//! through every module in the crate.

use std::rc::Rc;

use cubecl::prelude::Runtime;

use crate::backend::{Device, FloatElem};
use crate::error::{Error, Result};
use crate::tensor::{Shape, Tensor};

use super::graph::{BackwardRule, Graph, Grads, NodeId, ParamId};

/// Where a value sits on the tape.
pub(crate) struct Trace<R: Runtime, E: FloatElem> {
    pub(crate) graph: Rc<Graph<R, E>>,
    pub(crate) node: NodeId,
}

impl<R: Runtime, E: FloatElem> Clone for Trace<R, E> {
    fn clone(&self) -> Self {
        Self {
            graph: self.graph.clone(),
            node: self.node,
        }
    }
}

/// A tensor plus optional autodiff bookkeeping.
pub struct Var<R: Runtime, E: FloatElem = f32> {
    pub(crate) value: Tensor<R, E>,
    pub(crate) trace: Option<Trace<R, E>>,
}

impl<R: Runtime, E: FloatElem> Clone for Var<R, E> {
    fn clone(&self) -> Self {
        Self {
            value: self.value.clone(),
            trace: self.trace.clone(),
        }
    }
}

impl<R: Runtime, E: FloatElem> core::fmt::Debug for Var<R, E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "Var({}, tracked={})",
            self.value.shape(),
            self.trace.is_some()
        )
    }
}

impl<R: Runtime, E: FloatElem> Var<R, E> {
    /// Wrap a tensor without recording it. Gradients will not flow through it.
    pub fn constant(value: Tensor<R, E>) -> Self {
        Self { value, trace: None }
    }

    /// Wrap a tensor as a fresh tracked leaf on its own tape.
    ///
    /// Use this for model inputs when you want `d loss / d input`, or as the root
    /// of a standalone expression.
    pub fn traced(value: Tensor<R, E>) -> Self {
        let graph = Graph::new();
        let node = graph.push_leaf(None);
        Self {
            value,
            trace: Some(Trace { graph, node }),
        }
    }

    /// Wrap a tensor as a tracked parameter leaf on an existing tape.
    pub(crate) fn param_leaf(value: Tensor<R, E>, graph: Rc<Graph<R, E>>, id: ParamId) -> Self {
        let node = graph.push_leaf(Some(id));
        Self {
            value,
            trace: Some(Trace { graph, node }),
        }
    }

    /// The underlying tensor.
    pub fn tensor(&self) -> &Tensor<R, E> {
        &self.value
    }

    /// Consume the `Var`, returning the tensor.
    pub fn into_tensor(self) -> Tensor<R, E> {
        self.value
    }

    /// Shape of the value.
    pub fn shape(&self) -> &Shape {
        self.value.shape()
    }

    /// Dimensions of the value.
    pub fn dims(&self) -> &[usize] {
        self.value.dims()
    }

    /// Rank of the value.
    pub fn rank(&self) -> usize {
        self.value.rank()
    }

    /// Device the value lives on.
    pub fn device(&self) -> &Device<R> {
        self.value.device()
    }

    /// Download the value as `f32`.
    pub fn to_f32(&self) -> Vec<f32> {
        self.value.to_f32()
    }

    /// Whether this value is recorded on a tape.
    pub fn is_tracked(&self) -> bool {
        self.trace.is_some()
    }

    /// Drop the tape link, keeping the value. Gradients stop here.
    pub fn detach(&self) -> Self {
        Self {
            value: self.value.clone(),
            trace: None,
        }
    }

    /// The tape this value belongs to, if any.
    pub(crate) fn graph(&self) -> Option<Rc<Graph<R, E>>> {
        self.trace.as_ref().map(|t| t.graph.clone())
    }

    /// This value's node id, if tracked.
    pub fn node(&self) -> Option<NodeId> {
        self.trace.as_ref().map(|t| t.node)
    }

    /// Record an op with the given parents.
    ///
    /// If none of the parents are tracked the result is a plain constant and the
    /// rule is never built — that is the inference fast path.
    pub(crate) fn record(
        value: Tensor<R, E>,
        parents: &[&Var<R, E>],
        rule: impl FnOnce() -> BackwardRule<R, E>,
    ) -> Self {
        Self::record_with_mask(value, parents, |_| rule())
    }

    /// Record an op whose rule can be built cheaper when some parent does not want a
    /// gradient.
    ///
    /// The mask handed to the factory has one entry per parent: `true` where that
    /// parent is on the tape, `false` where it is a constant whose gradient would be
    /// discarded. A rule that ignores the mask is still correct — [`record`] is
    /// exactly that — but for a broadcasting binary op it is expensive to ignore.
    /// Multiplying by a constant mask is the case that matters here: the scan
    /// multiplies `[batch, chunks, chunk, chunk, heads]` intermediates by causal
    /// masks four times per layer, and the discarded gradient for each one was a
    /// full-size product followed by a three-axis reduction down to the mask's shape.
    ///
    /// [`record`]: Var::record
    pub(crate) fn record_with_mask(
        value: Tensor<R, E>,
        parents: &[&Var<R, E>],
        rule: impl FnOnce(&[bool]) -> BackwardRule<R, E>,
    ) -> Self {
        if !super::grad_mode::is_enabled() {
            return Var::constant(value);
        }
        let graph = match parents.iter().find_map(|p| p.graph()) {
            Some(g) => g,
            None => return Var::constant(value),
        };
        // Fold any other tapes into this one so a single traversal sees everything.
        for parent in parents {
            if let Some(other) = parent.graph()
                && !Rc::ptr_eq(&graph, &other)
            {
                graph.absorb(&other);
            }
        }
        let parent_ids: Vec<NodeId> = parents
            .iter()
            .map(|p| match p.node() {
                Some(id) => id,
                // Untracked parents still occupy a slot so the rule's output order
                // lines up with `parents`; their gradient is simply dropped.
                None => graph.push_leaf(None),
            })
            .collect();
        let wanted: Vec<bool> = parents.iter().map(|p| p.node().is_some()).collect();
        let node = graph.push(parent_ids, rule(&wanted));
        Self {
            value,
            trace: Some(Trace { graph, node }),
        }
    }

    /// Differentiate this value with respect to everything it depends on.
    ///
    /// The tape is cleared afterwards, releasing all saved activations. Use
    /// [`Var::backward_retain`] to keep it for a second pass.
    pub fn backward(&self) -> Result<Grads<R, E>> {
        let grads = self.backward_inner(false)?;
        if let Some(trace) = &self.trace {
            trace.graph.clear();
        }
        Ok(grads)
    }

    /// Like [`Var::backward`] but keeps the tape and every node gradient.
    pub fn backward_retain(&self) -> Result<Grads<R, E>> {
        self.backward_inner(true)
    }

    fn backward_inner(&self, retain_nodes: bool) -> Result<Grads<R, E>> {
        let trace = self.trace.as_ref().ok_or_else(|| {
            Error::Autodiff("backward() called on a value that is not tracked".into())
        })?;

        let mut pending: std::collections::HashMap<NodeId, Tensor<R, E>> =
            std::collections::HashMap::new();
        pending.insert(trace.node, Tensor::ones(self.value.shape().clone(), self.device()));

        let mut grads = Grads::default();

        trace.graph.with_nodes(|nodes| -> Result<()> {
            // Ids increase with creation order, so walking them downwards visits
            // every consumer before its producers.
            let mut ids: Vec<NodeId> = nodes.keys().copied().collect();
            ids.sort_unstable_by(|a, b| b.cmp(a));

            for id in ids {
                let Some(grad) = pending.remove(&id) else {
                    continue;
                };
                let Some(node) = nodes.get(&id) else {
                    continue;
                };
                if let Some(param) = node.param {
                    grads.accumulate(param, grad.clone())?;
                }
                if retain_nodes {
                    grads.set_node(id, grad.clone());
                }
                if node.parents.is_empty() {
                    continue;
                }
                let parent_grads = (node.rule)(&grad)?;
                for (parent, pg) in node.parents.iter().zip(parent_grads) {
                    let Some(pg) = pg else { continue };
                    match pending.remove(parent) {
                        Some(existing) => {
                            pending.insert(
                                *parent,
                                crate::tensor::ops::elemwise::add(&existing, &pg)?,
                            );
                        }
                        None => {
                            pending.insert(*parent, pg);
                        }
                    }
                }
            }
            Ok(())
        })?;

        Ok(grads)
    }
}

impl<R: Runtime, E: FloatElem> From<Tensor<R, E>> for Var<R, E> {
    fn from(value: Tensor<R, E>) -> Self {
        Var::constant(value)
    }
}
