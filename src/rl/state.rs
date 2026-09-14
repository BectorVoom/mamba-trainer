//! The persistent rollout state.
//!
//! A reinforcement learning rollout is the opposite of a training pass in what it
//! asks of memory. Training wants one big allocation, used once, and thrown away.
//! A rollout runs the same tiny computation millions of times and must not grow by
//! a byte while doing it, because the state is the *only* thing that persists: it
//! is what the environment's history has been compressed into.
//!
//! [`Mamba3StateBuffer`] is that state, allocated once for `B` environments and
//! `L` layers and then only ever overwritten. Its size is fixed by the
//! configuration — `B * L * (2 * heads * head_dim * d_state + conv)` elements —
//! and is the same at step 1 and step 1,000,000. That is the whole architectural
//! argument for a state space policy over a transformer one, stated as a number.

use cubecl::prelude::Runtime;

use crate::backend::{Device, FloatElem};
use crate::error::{Error, Result};
use crate::models::mamba3::MixerCache;
use crate::ssm::scan::SsmState;
use crate::tensor::Tensor;
use crate::tensor::ops::elemwise::clear_rows_;

/// Per-layer recurrent state for `B` environments, allocated once.
///
/// # In-place mutation
///
/// [`Mamba3StateBuffer::reset_env_states`] rewrites the buffers where they lie
/// rather than producing new ones. That is sound only because the buffer owns its
/// tensors exclusively: every entry is either a freshly zeroed allocation or the
/// output of the step that produced it, and nothing else holds a handle. The
/// escape hatch is [`Mamba3StateBuffer::snapshot`], which copies — hold on to one
/// of those, not to the buffer's own tensors, if you need a state to outlive the
/// next step.
pub struct Mamba3StateBuffer<R: Runtime, E: FloatElem> {
    layers: Vec<MixerCache<R, E>>,
    envs: usize,
    device: Device<R>,
}

impl<R: Runtime, E: FloatElem> Mamba3StateBuffer<R, E> {
    /// Allocate a zeroed buffer from per-layer caches.
    ///
    /// Prefer [`crate::rl::Mamba3Policy::empty_state`], which fills this in from
    /// the policy's own configuration.
    pub fn new(layers: Vec<MixerCache<R, E>>, envs: usize, device: &Device<R>) -> Self {
        Self {
            layers,
            envs,
            device: device.clone(),
        }
    }

    /// Number of environments the buffer holds state for.
    pub fn envs(&self) -> usize {
        self.envs
    }

    /// Number of layers.
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    /// Whether the buffer holds no layers.
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    /// The device the state lives on.
    pub fn device(&self) -> &Device<R> {
        &self.device
    }

    /// Elements held on the device.
    ///
    /// Constant for the life of the buffer: this is the number that does not grow
    /// with the number of steps taken.
    pub fn num_elements(&self) -> usize {
        self.layers.iter().map(|l| l.num_elements()).sum()
    }

    /// Bytes held on the device.
    pub fn bytes(&self) -> usize {
        self.num_elements() * core::mem::size_of::<E>()
    }

    /// Every layer's state, in order.
    pub fn layers(&self) -> &[MixerCache<R, E>] {
        &self.layers
    }

    /// Borrow one layer's state.
    pub fn layer(&self, index: usize) -> Result<&MixerCache<R, E>> {
        self.layers.get(index).ok_or_else(|| {
            Error::shape(format!(
                "layer {index} is out of range for a {}-layer state buffer",
                self.layers.len()
            ))
        })
    }

    /// Replace one layer's state with the result of a step.
    ///
    /// The incoming cache must have been produced from this buffer's own layer, so
    /// that the buffer keeps sole ownership of every tensor it holds.
    pub fn store(&mut self, index: usize, cache: MixerCache<R, E>) -> Result<()> {
        let len = self.layers.len();
        let slot = self.layers.get_mut(index).ok_or_else(|| {
            Error::shape(format!(
                "layer {index} is out of range for a {len}-layer state buffer"
            ))
        })?;
        *slot = cache;
        Ok(())
    }

    /// Independent copies of every layer's state, safe to keep across steps.
    pub fn snapshot(&self) -> Vec<MixerCache<R, E>> {
        self.layers
            .iter()
            .map(|l| MixerCache {
                ssm: SsmState {
                    h: crate::autograd::Var::constant(l.ssm.h.tensor().deep_clone()),
                    last_u: crate::autograd::Var::constant(l.ssm.last_u.tensor().deep_clone()),
                    angle: l
                        .ssm
                        .angle
                        .as_ref()
                        .map(|a| crate::autograd::Var::constant(a.tensor().deep_clone())),
                },
                conv: l
                    .conv
                    .as_ref()
                    .map(|c| crate::autograd::Var::constant(c.tensor().deep_clone())),
            })
            .collect()
    }

    /// Clear the state of terminated environments, in place.
    ///
    /// `mask` is `[envs]` (or any shape with `envs` elements), holding `1` for an
    /// environment whose episode ended and `0` for one that continues. Every
    /// layer's hidden state, trapezoidal carry and convolution history is zeroed
    /// for those rows and left untouched for the rest. Nothing is allocated, so
    /// this is safe to call on every step of an unbounded loop.
    ///
    /// # What is deliberately not reset
    ///
    /// The rotating frame's running angle survives. Only the *relative* rotation
    /// between a source position and the position reading it reaches the output —
    /// both are mapped by the same absolute frame, which cancels — so the angle is
    /// unobservable across a boundary, and zeroing the state above has already
    /// severed every pair that straddles one. Restarting it would be a pass over
    /// the angle table to produce identical numbers.
    ///
    /// # Relation to the fused path
    ///
    /// [`crate::rl::RolloutEngine::step`] does not call this. It hands the same
    /// mask to the step itself, where clearing the state costs nothing at all: the
    /// recurrence's two backward-looking coefficients are zeroed in the kernel
    /// that already computes them, instead of two passes over the state tensors.
    /// The two routes are numerically identical — `rollout_reset_matches_explicit_clear`
    /// in `tests/rl.rs` holds them to it. Use this one to reset outside a step, or
    /// when driving the layers by hand.
    pub fn reset_env_states(&mut self, mask: &Tensor<R, E>) -> Result<()> {
        if mask.len() != self.envs {
            return Err(Error::shape(format!(
                "reset mask must hold one flag per environment: expected {}, got {}",
                self.envs,
                mask.shape()
            )));
        }
        let mask = mask.reshape(vec![self.envs])?;
        for layer in &self.layers {
            clear_rows_(layer.ssm.h.tensor(), &mask)?;
            clear_rows_(layer.ssm.last_u.tensor(), &mask)?;
            if let Some(conv) = &layer.conv {
                clear_rows_(conv.tensor(), &mask)?;
            }
        }
        Ok(())
    }

    /// Zero every environment's state, in place.
    pub fn reset_all(&mut self) {
        for layer in &self.layers {
            crate::tensor::ops::elemwise::fill_(layer.ssm.h.tensor(), 0.0);
            crate::tensor::ops::elemwise::fill_(layer.ssm.last_u.tensor(), 0.0);
            if let Some(angle) = &layer.ssm.angle {
                crate::tensor::ops::elemwise::fill_(angle.tensor(), 0.0);
            }
            if let Some(conv) = &layer.conv {
                crate::tensor::ops::elemwise::fill_(conv.tensor(), 0.0);
            }
        }
    }
}

impl<R: Runtime, E: FloatElem> core::fmt::Debug for Mamba3StateBuffer<R, E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "Mamba3StateBuffer(envs={}, layers={}, {} elements, {} KiB)",
            self.envs,
            self.layers.len(),
            self.num_elements(),
            self.bytes() / 1024,
        )
    }
}
