//! Encoder/decoder blocks and index permutations (ENTITY_MODEL_PLAN.md G2).
//!
//! [`BiBlock`] and [`transpose_grid`] moved here verbatim from
//! `models::planner` (which re-imports them until G6 deletes it).
//! [`ForwardBlock`] is the same residual block with a forward-only mixer
//! (no head doubling) for [`crate::models::entity::DecoderMode::StepCausal`].
//! [`Permutation`] precomputes the whole-context index permutations (§2.1:
//! grid transposes) and the within-step-block reversals (§2.3,
//! `crew_symmetric`); the composed path applies them with [`Var::gather_tokens`],
//! the fused path with K4.

use cubecl::prelude::Runtime;

use crate::autograd::Var;
use crate::backend::{Device, FloatElem};
use crate::error::{Error, Result};
use crate::models::mamba3::{Mamba3Mixer, Mamba3MixerConfig};
use crate::nn::module::{Module, ModuleVisitor};
use crate::nn::norm::{RmsNorm, RmsNormConfig};
use crate::ssm::config::SsmConfig;
use crate::tensor::ops::index::IdTensor;
use crate::tensor::ops::random::Rng;

/// Build one residual mixer block's norm.
fn block_norm<R: Runtime, E: FloatElem>(
    d_model: usize,
    norm_eps: f32,
    device: &Device<R>,
    rng: &mut Rng,
) -> RmsNorm<R, E> {
    RmsNormConfig::new(d_model)
        .with_eps(norm_eps)
        .init(device, rng)
}

/// Build one mixer with `depth`, doubling heads/groups when bidirectional
/// (each direction keeps the configured capacity).
fn block_mixer<R: Runtime, E: FloatElem>(
    ssm: &SsmConfig,
    bidirectional: bool,
    depth: usize,
    device: &Device<R>,
    rng: &mut Rng,
) -> Result<Mamba3Mixer<R, E>> {
    let mut ssm = ssm.clone();
    if bidirectional {
        ssm.n_heads *= 2;
        ssm.n_groups *= 2;
    }
    Mamba3MixerConfig::new(ssm)
        .with_depth(depth)
        .with_bidirectional(bidirectional)
        .init(device, rng)
}

/// One residual bidirectional Mamba-3 block: `x + mixer(norm(x))`, where the
/// mixer is a single fused mixer whose second half of heads scans right to
/// left (as [`crate::models::vision::VisionMamba3`] does).
pub struct BiBlock<R: Runtime, E: FloatElem> {
    norm: RmsNorm<R, E>,
    mixer: Mamba3Mixer<R, E>,
}

impl<R: Runtime, E: FloatElem> BiBlock<R, E> {
    /// Build a block (heads/groups doubled for the fused bidirectional mixer).
    pub fn new(
        d_model: usize,
        ssm: &SsmConfig,
        norm_eps: f32,
        depth: usize,
        device: &Device<R>,
        rng: &mut Rng,
    ) -> Result<Self> {
        Ok(Self {
            norm: block_norm(d_model, norm_eps, device, rng),
            mixer: block_mixer(ssm, true, depth, device, rng)?,
        })
    }

    /// Apply the block to `[B, T, d]`.
    pub fn apply(&self, x: &Var<R, E>) -> Result<Var<R, E>> {
        x.add(&self.mixer.apply(&self.norm.apply(x)?)?)
    }
}

impl<R: Runtime, E: FloatElem> Module<R, E> for BiBlock<R, E> {
    fn visit(&self, visitor: &mut ModuleVisitor<'_, R, E>) {
        visitor.child("norm", &self.norm);
        visitor.child("mixer", &self.mixer);
    }
}

/// One residual forward-only Mamba-3 block: `x + mixer(norm(x))` with
/// [`Mamba3MixerConfig::with_bidirectional`] set to false, so a token sees
/// only what precedes it. No head doubling.
pub struct ForwardBlock<R: Runtime, E: FloatElem> {
    norm: RmsNorm<R, E>,
    mixer: Mamba3Mixer<R, E>,
}

impl<R: Runtime, E: FloatElem> ForwardBlock<R, E> {
    /// Build a forward-only block.
    pub fn new(
        d_model: usize,
        ssm: &SsmConfig,
        norm_eps: f32,
        depth: usize,
        device: &Device<R>,
        rng: &mut Rng,
    ) -> Result<Self> {
        Ok(Self {
            norm: block_norm(d_model, norm_eps, device, rng),
            mixer: block_mixer(ssm, false, depth, device, rng)?,
        })
    }

    /// Apply the block to `[B, T, d]`.
    pub fn apply(&self, x: &Var<R, E>) -> Result<Var<R, E>> {
        x.add(&self.mixer.apply(&self.norm.apply(x)?)?)
    }
}

impl<R: Runtime, E: FloatElem> Module<R, E> for ForwardBlock<R, E> {
    fn visit(&self, visitor: &mut ModuleVisitor<'_, R, E>) {
        visitor.child("norm", &self.norm);
        visitor.child("mixer", &self.mixer);
    }
}

/// `[B, g*g, d]` row-major <-> column-major. The same function converts both
/// ways (applying it twice is the identity).
pub fn transpose_grid<R: Runtime, E: FloatElem>(x: &Var<R, E>, g: usize) -> Result<Var<R, E>> {
    let (b, d) = (x.shape().dim(0), x.shape().dim(2));
    x.reshape(vec![b, g, g, d])?
        .permute(&[0, 2, 1, 3])?
        .reshape(vec![b, g * g, d])
}

/// A precomputed index permutation of a token sequence of length `n`.
///
/// Both constructors below are self-inverse (`inv == fwd`), so applying twice
/// is the identity; `inv` is still stored for the K4 kernel contract.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Permutation {
    /// `out[i] = src[fwd[i]]`.
    pub fwd: Vec<u32>,
    /// `out[i] = src[inv[i]]` undoes `fwd`.
    pub inv: Vec<u32>,
}

impl Permutation {
    /// The identity permutation of length `n`.
    pub fn identity(n: usize) -> Self {
        let fwd: Vec<u32> = (0..n as u32).collect();
        Self {
            fwd: fwd.clone(),
            inv: fwd,
        }
    }

    /// Whether this is the identity (applying it is a no-op copy).
    pub fn is_identity(&self) -> bool {
        self.fwd.iter().enumerate().all(|(i, &v)| v as usize == i)
    }

    /// Length of the permuted sequence.
    pub fn len(&self) -> usize {
        self.fwd.len()
    }

    /// Whether the sequence is empty.
    pub fn is_empty(&self) -> bool {
        self.fwd.is_empty()
    }

    /// Build from a forward index table (`out[i] = src[fwd[i]]`); the
    /// inverse is derived automatically.
    pub fn from_fwd(fwd: Vec<u32>) -> Self {
        let mut inv = vec![0u32; fwd.len()];
        for (i, &v) in fwd.iter().enumerate() {
            inv[v as usize] = i as u32;
        }
        Self { fwd, inv }
    }

    /// Transpose the `h × w` grid starting at `offset` (row-major <->
    /// column-major), identity elsewhere. Self-inverse.
    pub fn grid_transpose(offset: usize, h: usize, w: usize, n: usize) -> Self {
        assert!(h * w + offset <= n);
        let mut fwd: Vec<u32> = (0..n as u32).collect();
        for r in 0..h {
            for c in 0..w {
                fwd[offset + c * h + r] = (offset + r * w + c) as u32;
            }
        }
        Self {
            fwd: fwd.clone(),
            inv: fwd,
        }
    }

    /// Reverse the tokens inside each of `blocks` consecutive blocks of
    /// `block` tokens starting at `offset`, identity elsewhere. Self-inverse.
    pub fn reverse_blocks(offset: usize, block: usize, blocks: usize, n: usize) -> Self {
        assert!(offset + block * blocks <= n);
        let mut fwd: Vec<u32> = (0..n as u32).collect();
        for k in 0..blocks {
            for i in 0..block {
                fwd[offset + k * block + i] = (offset + k * block + (block - 1 - i)) as u32;
            }
        }
        Self {
            fwd: fwd.clone(),
            inv: fwd,
        }
    }

    /// Apply to `[B, n, d]` on the composed path (gather of rows by index).
    pub fn apply<R: Runtime, E: FloatElem>(&self, x: &Var<R, E>) -> Result<Var<R, E>> {
        let dims = x.shape().dims().to_vec();
        if dims.len() != 3 || dims[1] != self.len() {
            return Err(Error::shape(format!(
                "permutation of length {} does not fit {}",
                self.len(),
                x.shape()
            )));
        }
        let (b, n) = (dims[0], dims[1]);
        let mut ids = Vec::with_capacity(b * n);
        for _ in 0..b {
            ids.extend_from_slice(&self.fwd);
        }
        let ids = IdTensor::from_slice(&ids, vec![b * n], x.device())?;
        Var::gather_tokens(x, &ids, n)
    }

    /// Undo [`Permutation::apply`].
    pub fn inverse<R: Runtime, E: FloatElem>(&self, x: &Var<R, E>) -> Result<Var<R, E>> {
        Self {
            fwd: self.inv.clone(),
            inv: self.fwd.clone(),
        }
        .apply(x)
    }
}

/// A decoder layer: one main block plus, for `StepCausal { crew_symmetric:
/// true }`, a second forward-only block run on the same sequence with the
/// query tokens reversed inside each step block (another precomputed
/// permutation), un-permuted and added.
pub struct DecoderLayer<R: Runtime, E: FloatElem> {
    /// Main block: bidirectional in `Joint` mode, forward-only in `StepCausal`.
    pub main_bi: Option<BiBlock<R, E>>,
    /// Main forward-only block (`StepCausal` mode).
    pub main_fwd: Option<ForwardBlock<R, E>>,
    /// Second forward-only scan for `crew_symmetric` (`StepCausal` only).
    pub rev: Option<ForwardBlock<R, E>>,
    /// Within-step-block reversal (length = decoder sequence); `None` without
    /// `crew_symmetric`.
    pub rev_perm: Option<Permutation>,
}

impl<R: Runtime, E: FloatElem> DecoderLayer<R, E> {
    /// A `Joint`-mode layer (bidirectional main block, no second scan).
    pub fn joint(
        d_model: usize,
        ssm: &SsmConfig,
        norm_eps: f32,
        depth: usize,
        device: &Device<R>,
        rng: &mut Rng,
    ) -> Result<Self> {
        Ok(Self {
            main_bi: Some(BiBlock::new(d_model, ssm, norm_eps, depth, device, rng)?),
            main_fwd: None,
            rev: None,
            rev_perm: None,
        })
    }

    /// A `StepCausal`-mode layer, with or without the `crew_symmetric` scan.
    #[allow(clippy::too_many_arguments)]
    pub fn step_causal(
        d_model: usize,
        ssm: &SsmConfig,
        norm_eps: f32,
        depth: usize,
        crew_symmetric: bool,
        rev_perm: Option<Permutation>,
        device: &Device<R>,
        rng: &mut Rng,
    ) -> Result<Self> {
        let rev = if crew_symmetric {
            Some(ForwardBlock::new(
                d_model, ssm, norm_eps, depth, device, rng,
            )?)
        } else {
            None
        };
        Ok(Self {
            main_bi: None,
            main_fwd: Some(ForwardBlock::new(
                d_model, ssm, norm_eps, depth, device, rng,
            )?),
            rev,
            rev_perm,
        })
    }

    /// Apply to the decoder sequence `[B, T, d]`. `rev_dev` carries the
    /// device-side within-step reversal (`None` = composed path with the
    /// host [`Permutation`]).
    pub fn apply(
        &self,
        x: &Var<R, E>,
        rev_dev: Option<(&IdTensor<R>, &IdTensor<R>)>,
    ) -> Result<Var<R, E>> {
        let mut y = if let Some(b) = &self.main_bi {
            b.apply(x)?
        } else if let Some(b) = &self.main_fwd {
            b.apply(x)?
        } else {
            return Err(Error::config("decoder layer has no main block".to_string()));
        };
        if let (Some(rev), Some(perm)) = (&self.rev, &self.rev_perm) {
            let r = match rev_dev {
                Some((fwd, inv)) => Var::permute_tokens(x, fwd, inv)?,
                None => perm.apply(x)?,
            };
            let r = rev.apply(&r)?;
            let r = match rev_dev {
                Some((fwd, inv)) => Var::permute_tokens(&r, inv, fwd)?,
                None => perm.inverse(&r)?,
            };
            y = y.add(&r.sub(x)?)?;
        }
        Ok(y)
    }
}

impl<R: Runtime, E: FloatElem> Module<R, E> for DecoderLayer<R, E> {
    fn visit(&self, visitor: &mut ModuleVisitor<'_, R, E>) {
        if let Some(b) = &self.main_bi {
            visitor.child("main", b);
        }
        if let Some(b) = &self.main_fwd {
            visitor.child("main", b);
        }
        if let Some(b) = &self.rev {
            visitor.child("rev", b);
        }
    }
}
