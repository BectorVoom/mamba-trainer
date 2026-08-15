//! The module tree: parameter discovery, mode switching, and state dicts.
//!
//! A module describes its own structure once, in [`Module::visit`]. Everything
//! else — named parameters, parameter counts, freezing, train/eval switching,
//! checkpoint save and load — is a traversal of that single description, so a new
//! layer only ever writes one method.

use std::collections::BTreeMap;

use cubecl::prelude::Runtime;

use crate::backend::FloatElem;
use crate::error::{Error, Result};
use crate::nn::param::Param;
use crate::tensor::{Shape, Tensor};

/// What a traversal is doing at each node.
enum Action<'a, R: Runtime, E: FloatElem> {
    /// Report every parameter with its dotted path.
    Params(&'a mut dyn FnMut(String, &Param<R, E>)),
    /// Switch train/eval mode.
    Training(bool),
    /// Fold LoRA adapters into their base weights, recording the first failure.
    MergeLora(&'a mut Option<Error>),
}

/// Carries the current path while walking a module tree.
pub struct ModuleVisitor<'a, R: Runtime, E: FloatElem> {
    path: Vec<String>,
    action: Action<'a, R, E>,
}

impl<'a, R: Runtime, E: FloatElem> ModuleVisitor<'a, R, E> {
    /// Report a parameter owned directly by the current module.
    pub fn param(&mut self, name: &str, param: &Param<R, E>) {
        let full = self.join(name);
        if let Action::Params(sink) = &mut self.action {
            sink(full, param);
        }
    }

    /// Descend into a child module.
    pub fn child<M: Module<R, E> + ?Sized>(&mut self, name: &str, module: &M) {
        match &mut self.action {
            Action::Training(flag) => module.on_mode_change(*flag),
            Action::MergeLora(slot) => {
                if let Err(err) = module.on_merge_lora()
                    && slot.is_none()
                {
                    **slot = Some(err);
                }
            }
            Action::Params(_) => {}
        }
        self.path.push(name.to_string());
        module.visit(self);
        self.path.pop();
    }

    /// Descend into an element of a list of children, e.g. `blocks.3`.
    pub fn child_at<M: Module<R, E> + ?Sized>(&mut self, name: &str, index: usize, module: &M) {
        self.child(&format!("{name}.{index}"), module);
    }

    /// Descend into an optional child.
    pub fn child_opt<M: Module<R, E>>(&mut self, name: &str, module: &Option<M>) {
        if let Some(m) = module {
            self.child(name, m);
        }
    }

    fn join(&self, name: &str) -> String {
        if self.path.is_empty() {
            name.to_string()
        } else {
            format!("{}.{}", self.path.join("."), name)
        }
    }
}

/// A node in the model tree.
///
/// Implementors describe their parameters and children in [`Module::visit`]. Every
/// other method has a working default.
pub trait Module<R: Runtime, E: FloatElem> {
    /// Announce this module's own parameters and children.
    fn visit(&self, visitor: &mut ModuleVisitor<'_, R, E>);

    /// Hook called on **this** module when the train/eval mode changes.
    ///
    /// Children are handled by the traversal; only stateful layers such as dropout
    /// need to override this.
    fn on_mode_change(&self, _training: bool) {}

    /// Hook called on **this** module when adapters are being merged.
    ///
    /// Only layers that own a LoRA adapter need to override it.
    fn on_merge_lora(&self) -> Result<()> {
        Ok(())
    }

    /// Every parameter with its dotted path, in a deterministic order.
    fn named_parameters(&self) -> Vec<(String, Param<R, E>)>
    where
        Self: Sized,
    {
        let mut out = Vec::new();
        {
            let mut sink = |name: String, param: &Param<R, E>| out.push((name, param.clone()));
            let mut visitor = ModuleVisitor {
                path: Vec::new(),
                action: Action::Params(&mut sink),
            };
            self.visit(&mut visitor);
        }
        out
    }

    /// Every parameter, without names.
    fn parameters(&self) -> Vec<Param<R, E>>
    where
        Self: Sized,
    {
        self.named_parameters().into_iter().map(|(_, p)| p).collect()
    }

    /// Only the parameters that will receive gradients.
    fn trainable_parameters(&self) -> Vec<Param<R, E>>
    where
        Self: Sized,
    {
        self.parameters()
            .into_iter()
            .filter(|p| p.requires_grad())
            .collect()
    }

    /// Total number of scalars in the model.
    fn num_parameters(&self) -> usize
    where
        Self: Sized,
    {
        self.parameters().iter().map(|p| p.numel()).sum()
    }

    /// Number of scalars that are currently being trained.
    fn num_trainable_parameters(&self) -> usize
    where
        Self: Sized,
    {
        self.trainable_parameters().iter().map(|p| p.numel()).sum()
    }

    /// Switch the whole subtree between training and evaluation behaviour.
    fn set_training(&self, training: bool)
    where
        Self: Sized,
    {
        self.on_mode_change(training);
        let mut visitor = ModuleVisitor {
            path: Vec::new(),
            action: Action::Training(training),
        };
        self.visit(&mut visitor);
    }

    /// Shorthand for `set_training(true)`.
    fn train(&self)
    where
        Self: Sized,
    {
        self.set_training(true);
    }

    /// Shorthand for `set_training(false)`.
    fn eval(&self)
    where
        Self: Sized,
    {
        self.set_training(false);
    }

    /// Fold every LoRA adapter in the subtree into its base weight.
    ///
    /// After this the adapters contribute nothing until they are trained again, so
    /// the call is idempotent and inference pays no adapter cost.
    fn merge_lora_adapters(&self) -> Result<()>
    where
        Self: Sized,
    {
        self.on_merge_lora()?;
        let mut failure = None;
        {
            let mut visitor = ModuleVisitor {
                path: Vec::new(),
                action: Action::MergeLora(&mut failure),
            };
            self.visit(&mut visitor);
        }
        match failure {
            Some(err) => Err(err),
            None => Ok(()),
        }
    }

    /// Freeze every parameter whose path contains any of `patterns`. With an empty
    /// pattern list, freeze everything.
    fn freeze_matching(&self, patterns: &[&str])
    where
        Self: Sized,
    {
        for (name, param) in self.named_parameters() {
            if patterns.is_empty() || patterns.iter().any(|p| name.contains(p)) {
                param.freeze();
            }
        }
    }

    /// Unfreeze every parameter whose path contains any of `patterns`.
    fn unfreeze_matching(&self, patterns: &[&str])
    where
        Self: Sized,
    {
        for (name, param) in self.named_parameters() {
            if patterns.is_empty() || patterns.iter().any(|p| name.contains(p)) {
                param.set_requires_grad(true);
            }
        }
    }

    /// Export weights as host-side arrays, keyed by parameter path.
    fn state_dict(&self) -> StateDict
    where
        Self: Sized,
    {
        let mut out = BTreeMap::new();
        for (name, param) in self.named_parameters() {
            let value = param.value();
            out.insert(
                name,
                TensorData {
                    shape: value.shape().dims().to_vec(),
                    data: value.to_f32(),
                },
            );
        }
        StateDict { entries: out }
    }

    /// Load weights previously produced by [`Module::state_dict`].
    ///
    /// `strict` requires an exact key match in both directions.
    fn load_state_dict(&self, state: &StateDict, strict: bool) -> Result<()>
    where
        Self: Sized,
    {
        let params = self.named_parameters();
        if strict {
            for (name, _) in &params {
                if !state.entries.contains_key(name) {
                    return Err(Error::StateDict(format!("missing entry for `{name}`")));
                }
            }
            for name in state.entries.keys() {
                if !params.iter().any(|(n, _)| n == name) {
                    return Err(Error::StateDict(format!("unexpected entry `{name}`")));
                }
            }
        }
        for (name, param) in params {
            let Some(entry) = state.entries.get(&name) else {
                continue;
            };
            let expected = param.shape();
            if entry.shape != expected.dims() {
                return Err(Error::StateDict(format!(
                    "`{name}`: checkpoint holds {:?} but the model expects {expected}",
                    entry.shape
                )));
            }
            let tensor = Tensor::<R, E>::from_f32(
                &entry.data,
                Shape::new(entry.shape.clone()),
                &param.value().device().clone(),
            )?;
            param.set(tensor);
        }
        Ok(())
    }

    /// A one-line-per-parameter summary, useful in examples and tests.
    fn describe(&self) -> String
    where
        Self: Sized,
    {
        let mut out = String::new();
        for (name, param) in self.named_parameters() {
            out.push_str(&format!(
                "{name:<48} {:>14} {:>10}{}\n",
                param.shape().to_string(),
                param.numel(),
                if param.requires_grad() { "" } else { "  (frozen)" }
            ));
        }
        out.push_str(&format!(
            "total {} parameters, {} trainable\n",
            self.num_parameters(),
            self.num_trainable_parameters()
        ));
        out
    }
}

/// A module that maps one value to one value.
///
/// Implementing this makes a layer usable inside [`Sequential`] and anywhere a
/// generic block is expected.
pub trait Layer<R: Runtime, E: FloatElem>: Module<R, E> {
    /// Apply the layer.
    fn forward(&self, input: &crate::autograd::Var<R, E>)
    -> Result<crate::autograd::Var<R, E>>;
}

/// One tensor's worth of host-side weights.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TensorData {
    /// Dimensions.
    pub shape: Vec<usize>,
    /// Row-major values.
    pub data: Vec<f32>,
}

/// A portable snapshot of a model's weights.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct StateDict {
    /// Parameter path to values.
    pub entries: BTreeMap<String, TensorData>,
}

impl StateDict {
    /// Write as JSON.
    pub fn save(&self, path: impl AsRef<std::path::Path>) -> Result<()> {
        let file = std::fs::File::create(path)?;
        serde_json::to_writer(std::io::BufWriter::new(file), self)?;
        Ok(())
    }

    /// Read from JSON.
    pub fn load(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let file = std::fs::File::open(path)?;
        Ok(serde_json::from_reader(std::io::BufReader::new(file))?)
    }

    /// Keep only entries whose key contains `pattern`. Used to ship LoRA-only
    /// checkpoints.
    pub fn filter(&self, pattern: &str) -> StateDict {
        StateDict {
            entries: self
                .entries
                .iter()
                .filter(|(k, _)| k.contains(pattern))
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
        }
    }
}

/// A chain of uniform layers.
pub struct Sequential<R: Runtime, E: FloatElem> {
    layers: Vec<Box<dyn Layer<R, E>>>,
}

impl<R: Runtime, E: FloatElem> Default for Sequential<R, E> {
    fn default() -> Self {
        Self { layers: Vec::new() }
    }
}

impl<R: Runtime, E: FloatElem> Sequential<R, E> {
    /// An empty chain.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a layer.
    pub fn push(mut self, layer: impl Layer<R, E> + 'static) -> Self {
        self.layers.push(Box::new(layer));
        self
    }

    /// Number of layers.
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    /// Whether the chain is empty.
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }
}

impl<R: Runtime, E: FloatElem> Module<R, E> for Sequential<R, E> {
    fn visit(&self, visitor: &mut ModuleVisitor<'_, R, E>) {
        for (i, layer) in self.layers.iter().enumerate() {
            visitor.child_at("layers", i, layer.as_ref());
        }
    }
}

impl<R: Runtime, E: FloatElem> Layer<R, E> for Sequential<R, E> {
    fn forward(
        &self,
        input: &crate::autograd::Var<R, E>,
    ) -> Result<crate::autograd::Var<R, E>> {
        let mut current = input.clone();
        for layer in &self.layers {
            current = layer.forward(&current)?;
        }
        Ok(current)
    }
}
