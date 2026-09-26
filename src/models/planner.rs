//! Task planner: next K work visits per unit over a tile grid, all Mamba-3 (see TASK_PLANNER_PLAN.md).

use cubecl::prelude::Runtime;

use crate::autograd::Var;
use crate::backend::{Device, FloatElem};
use crate::error::{Error, Result};
use crate::models::mamba3::{Mamba3Mixer, Mamba3MixerConfig};
use crate::nn::linear::{Linear, LinearConfig};
use crate::nn::module::{Module, ModuleVisitor};
use crate::nn::norm::{RmsNorm, RmsNormConfig};
use crate::nn::param::Param;
use crate::tensor::Tensor;
use crate::tensor::ops::index::IdTensor;
use crate::tensor::ops::random::Rng;
use crate::ssm::config::SsmConfig;

// --- fused-path switch (K0) -------------------------------------------------
//
// The composed path below is the correctness oracle. K1–K5 add on-device
// kernels behind this switch; until they land, every call site takes the
// composed path regardless of the flag.

/// Whether the fused on-device planner path is used: `0` off, `1` on, `-1`
/// not yet read from the environment.
static FUSED_PLANNER: core::sync::atomic::AtomicI8 = core::sync::atomic::AtomicI8::new(-1);

/// Whether the fused on-device planner path is enabled.
///
/// On by default, and `MAMBA3_FUSED_PLANNER=0` turns it off.
pub(crate) fn fused_planner_enabled() -> bool {
    use core::sync::atomic::Ordering;
    match FUSED_PLANNER.load(Ordering::Relaxed) {
        -1 => {
            let on = std::env::var("MAMBA3_FUSED_PLANNER").as_deref() != Ok("0");
            FUSED_PLANNER.store(on as i8, Ordering::Relaxed);
            on
        }
        flag => flag == 1,
    }
}

/// Choose whether the planner uses its fused on-device kernels or the composed
/// oracle path. Both produce the same tensors; this changes only how many
/// dispatches a step costs.
pub fn set_fused_planner(on: bool) {
    FUSED_PLANNER.store(on as i8, core::sync::atomic::Ordering::Relaxed);
}

/// Whether the fused on-device planner path is currently enabled.
pub fn fused_planner() -> bool {
    fused_planner_enabled()
}

/// Configuration for [`TaskPlanner`](crate::models::planner::TaskPlanner).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct TaskPlannerConfig {
    /// Grid edge length (tiles = grid * grid).
    pub grid: usize,
    /// Per-tile feature width.
    pub c_tile: usize,
    /// Global feature width.
    pub c_glob: usize,
    /// Per-unit feature width.
    pub c_unit: usize,
    /// Crew slots (always padded to this).
    pub max_units: usize,
    /// Work visits predicted per unit.
    pub k: usize,
    /// Number of op classes.
    pub n_ops: usize,
    /// Number of crop classes.
    pub n_crops: usize,
    /// Residual stream width.
    pub d_model: usize,
    /// Tile-mixer layers.
    pub n_tile_layers: usize,
    /// Joint-mixer layers.
    pub n_joint_layers: usize,
    /// Alternate row-major / column-major tile order on odd layers.
    pub alternate_axes: bool,
    /// Mixer settings (with `d_model` filled in at validation/init).
    pub ssm: SsmConfig,
    /// Normalisation epsilon.
    pub norm_eps: f32,
    /// Initialisation seed.
    pub seed: u64,
}

impl Default for TaskPlannerConfig {
    fn default() -> Self {
        let ssm = SsmConfig {
            d_model: 128,
            n_heads: 4,
            head_dim: 64,
            d_state: 32,
            n_groups: 1,
            chunk_size: 32,
            ..SsmConfig::default()
        };
        Self {
            grid: 10,
            c_tile: 48,
            c_glob: 114,
            c_unit: 36,
            max_units: 20,
            k: 3,
            n_ops: 13,
            n_crops: 5,
            d_model: 128,
            n_tile_layers: 3,
            n_joint_layers: 3,
            alternate_axes: true,
            ssm,
            norm_eps: 1e-5,
            seed: 0,
        }
    }
}

impl TaskPlannerConfig {
    /// Number of tiles (`grid * grid`).
    pub fn tiles(&self) -> usize {
        self.grid * self.grid
    }

    /// Number of query tokens (`max_units * k`).
    pub fn queries(&self) -> usize {
        self.max_units * self.k
    }

    /// Width of the fused aux head (`n_ops op | n_ops opset | n_crops crop | 1 eta`).
    pub fn aux_width(&self) -> usize {
        self.n_ops + self.n_ops + self.n_crops + 1
    }

    /// Check internal consistency.
    pub fn validate(&self) -> Result<()> {
        if self.grid == 0
            || self.c_tile == 0
            || self.c_glob == 0
            || self.c_unit == 0
            || self.max_units == 0
            || self.k == 0
            || self.n_ops == 0
            || self.n_crops == 0
            || self.d_model == 0
        {
            return Err(Error::config(
                "planner sizes (grid, features, units, k, classes, d_model) must all be positive",
            ));
        }
        if self.n_tile_layers == 0 || self.n_joint_layers == 0 {
            return Err(Error::config(
                "planner n_tile_layers and n_joint_layers must be positive",
            ));
        }
        if !(self.norm_eps > 0.0) {
            return Err(Error::config("planner norm_eps must be positive"));
        }
        let mut ssm = self.ssm.clone();
        ssm.d_model = self.d_model;
        ssm.validate()?;
        Ok(())
    }

    /// Instantiate with the configured seed.
    pub fn init<R: Runtime, E: FloatElem>(
        &self,
        device: &Device<R>,
    ) -> Result<TaskPlanner<R, E>> {
        let mut rng = Rng::seeded(self.seed);
        self.init_with_rng(device, &mut rng)
    }

    /// Instantiate with an explicit RNG.
    pub fn init_with_rng<R: Runtime, E: FloatElem>(
        &self,
        device: &Device<R>,
        rng: &mut Rng,
    ) -> Result<TaskPlanner<R, E>> {
        self.validate()?;
        let d = self.d_model;
        let param = |rows: usize, device: &Device<R>, rng: &mut Rng| -> Result<Param<R, E>> {
            Ok(Param::new(Tensor::from_f32(
                &rng.normal_vec(rows * d, 0.0, 0.02),
                vec![rows, d],
                device,
            )?))
        };
        let mut tile_blocks = Vec::with_capacity(self.n_tile_layers);
        for _ in 0..self.n_tile_layers {
            tile_blocks.push(BiBlock::new(self, self.n_tile_layers + self.n_joint_layers, device, rng)?);
        }
        let mut joint_blocks = Vec::with_capacity(self.n_joint_layers);
        for _ in 0..self.n_joint_layers {
            joint_blocks.push(BiBlock::new(self, self.n_tile_layers + self.n_joint_layers, device, rng)?);
        }
        Ok(TaskPlanner {
            tile_in1: LinearConfig::new(self.c_tile, d).init(device, rng),
            tile_in2: LinearConfig::new(d, d).init(device, rng),
            glob_in1: LinearConfig::new(self.c_glob, d).init(device, rng),
            glob_in2: LinearConfig::new(d, d).init(device, rng),
            unit_in1: LinearConfig::new(self.c_unit, d).init(device, rng),
            unit_in2: LinearConfig::new(d, d).init(device, rng),
            tile_pos: param(self.tiles(), device, rng)?,
            step_emb: param(self.k, device, rng)?,
            none_key: Param::new(Tensor::from_f32(
                &rng.normal_vec(d, 0.0, 0.02),
                vec![1, d],
                device,
            )?),
            tile_blocks,
            joint_blocks,
            norm: RmsNormConfig::new(d).with_eps(self.norm_eps).init(device, rng),
            q_proj: LinearConfig::new(d, d).init(device, rng),
            k_proj: LinearConfig::new(d, d).init(device, rng),
            aux: LinearConfig::new(2 * d, self.aux_width()).init(device, rng),
            config: self.clone(),
        })
    }
}

/// One residual bidirectional Mamba-3 block: `x + mixer(norm(x))`, where the
/// mixer is a single fused mixer whose second half of heads scans right to
/// left (as [`VisionMamba3`](crate::models::vision::VisionMamba3) does).
pub struct BiBlock<R: Runtime, E: FloatElem> {
    norm: RmsNorm<R, E>,
    mixer: Mamba3Mixer<R, E>,
}

impl<R: Runtime, E: FloatElem> BiBlock<R, E> {
    /// Build a block for `cfg` (heads/groups doubled for the fused bidirectional mixer).
    pub fn new(
        cfg: &TaskPlannerConfig,
        depth: usize,
        device: &Device<R>,
        rng: &mut Rng,
    ) -> Result<Self> {
        let mut ssm = cfg.ssm.clone();
        ssm.d_model = cfg.d_model;
        // Bidirectional: each direction keeps the configured capacity.
        ssm.n_heads *= 2;
        ssm.n_groups *= 2;
        Ok(Self {
            norm: RmsNormConfig::new(cfg.d_model)
                .with_eps(cfg.norm_eps)
                .init(device, rng),
            mixer: Mamba3MixerConfig::new(ssm)
                .with_depth(depth)
                .with_bidirectional(true)
                .init(device, rng)?,
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

/// `[B, g*g, d]` row-major <-> column-major. The same function converts both
/// ways (applying it twice is the identity).
pub fn transpose_grid<R: Runtime, E: FloatElem>(x: &Var<R, E>, g: usize) -> Result<Var<R, E>> {
    let (b, d) = (x.shape().dim(0), x.shape().dim(2));
    x.reshape(vec![b, g, g, d])?
        .permute(&[0, 2, 1, 3])?
        .reshape(vec![b, g * g, d])
}

/// The task planner: tile mixer, unit queries, joint mixer, pointer head.
pub struct TaskPlanner<R: Runtime, E: FloatElem> {
    tile_in1: Linear<R, E>,
    tile_in2: Linear<R, E>,
    glob_in1: Linear<R, E>,
    glob_in2: Linear<R, E>,
    unit_in1: Linear<R, E>,
    unit_in2: Linear<R, E>,
    tile_pos: Param<R, E>,
    step_emb: Param<R, E>,
    none_key: Param<R, E>,
    tile_blocks: Vec<BiBlock<R, E>>,
    joint_blocks: Vec<BiBlock<R, E>>,
    norm: RmsNorm<R, E>,
    q_proj: Linear<R, E>,
    k_proj: Linear<R, E>,
    aux: Linear<R, E>,
    config: TaskPlannerConfig,
}

/// Output of [`TaskPlanner::forward`].
pub struct PlannerOutput<R: Runtime, E: FloatElem> {
    /// `[B, Q, tiles + 1]` pointer logits over tiles plus NONE.
    pub target_logits: Var<R, E>,
    /// `[B, Q, d]` updated query tokens (H).
    pub queries: Var<R, E>,
    /// `[B, tiles + 1, d]` updated tile tokens plus the NONE key, before `k_proj`.
    pub keys_full: Var<R, E>,
}

impl<R: Runtime, E: FloatElem> TaskPlanner<R, E> {
    /// The configuration this model was built from.
    pub fn config(&self) -> &TaskPlannerConfig {
        &self.config
    }

    /// Run the tile mixer with row / column alternation.
    fn tile_mix(&self, t: &Var<R, E>) -> Result<Var<R, E>> {
        let g = self.config.grid;
        let mut t = t.clone();
        for (i, block) in self.tile_blocks.iter().enumerate() {
            if self.config.alternate_axes && i % 2 == 1 {
                t = transpose_grid(&t, g)?;
                t = block.apply(&t)?;
                t = transpose_grid(&t, g)?;
            } else {
                t = block.apply(&t)?;
            }
        }
        Ok(t)
    }

    /// Forward pass producing target logits.
    ///
    /// `unit_onehot` is `[B, U, N]` (padding rows are all zero). The batch must
    /// always be padded to exactly `max_units`: padding tokens still advance
    /// the SSM state, so trimming them would change train/inference behaviour.
    pub fn forward(
        &self,
        tiles: &Var<R, E>,
        glob: &Var<R, E>,
        units: &Var<R, E>,
        unit_onehot: &Tensor<R, E>,
    ) -> Result<PlannerOutput<R, E>> {
        let cfg = &self.config;
        let (b, d, n, u, k, q) = (
            tiles.shape().dim(0),
            cfg.d_model,
            cfg.tiles(),
            cfg.max_units,
            cfg.k,
            cfg.queries(),
        );
        // 1-4: embed tiles, add position and the broadcast global features.
        let mut t = self
            .tile_in2
            .apply(&self.tile_in1.apply(tiles)?.gelu()?)?;
        t = t.add(&self.tile_pos.var(&t).expand(vec![b, n, d])?)?;
        let g = self.glob_in2.apply(&self.glob_in1.apply(glob)?.gelu()?)?;
        t = t.add(&g.reshape(vec![b, 1, d])?.expand(vec![b, n, d])?)?;
        // 5: tile mixer.
        t = self.tile_mix(&t)?;
        // 6-8: the unit's own tile token plus the unit MLP output.
        let here = Var::constant(unit_onehot.clone()).matmul(&t)?;
        let um = self
            .unit_in2
            .apply(&self.unit_in1.apply(units)?.gelu()?)?;
        let uvar = um.add(&here)?;
        // 9: queries, unit-major (`q = u*K + j`).
        let step = self
            .step_emb
            .var(&uvar)
            .reshape(vec![1, 1, k, d])?
            .expand(vec![b, u, k, d])?;
        let qvar = uvar
            .reshape(vec![b, u, 1, d])?
            .expand(vec![b, u, k, d])?
            .add(&step)?
            .reshape(vec![b, q, d])?;
        // 10-12: joint sequence through the joint mixer and the final norm.
        let mut s = crate::autograd::cat(&[t, qvar], 1)?;
        for block in &self.joint_blocks {
            s = block.apply(&s)?;
        }
        s = self.norm.apply(&s)?;
        // 13-15: split, append the NONE key, score with one batched matmul_nt.
        let t2 = s.slice(1, 0, n)?;
        let h = s.slice(1, n, q)?;
        let none = self
            .none_key
            .var(&s)
            .reshape(vec![1, 1, d])?
            .expand(vec![b, 1, d])?;
        let keys_full = crate::autograd::cat(&[t2, none], 1)?;
        let logits = self
            .q_proj
            .apply(&h)?
            .matmul_nt(&self.k_proj.apply(&keys_full)?)?
            .mul_scalar(1.0 / (d as f32).sqrt());
        Ok(PlannerOutput {
            target_logits: logits,
            queries: h,
            keys_full,
        })
    }

    /// Aux head outputs at the given targets: `tt = one_hot(target) @ keys`
    /// then one Linear over `concat(H, tt)`. Returns `[B, Q, aux_width]`
    /// (`op | opset | crop | eta`); split only in the loss / on the host.
    pub fn heads(
        &self,
        out: &PlannerOutput<R, E>,
        target_onehot: &Tensor<R, E>,
    ) -> Result<Var<R, E>> {
        let tt = Var::constant(target_onehot.clone()).matmul(&out.keys_full)?;
        self.aux
            .apply(&crate::autograd::cat(&[out.queries.clone(), tt], 2)?)
    }

    /// Save weights plus the config (as metadata) to `path`.
    pub fn save(&self, path: impl AsRef<std::path::Path>, step: u64) -> Result<()> {
        crate::train::Checkpoint::capture(self, step)
            .with_metadata(serde_json::to_value(&self.config)?)
            .save(path)
    }

    /// Rebuild from a checkpoint's metadata and restore its weights.
    pub fn load(path: impl AsRef<std::path::Path>, device: &Device<R>) -> Result<Self> {
        let ckpt = crate::train::Checkpoint::load(path)?;
        let config: TaskPlannerConfig = serde_json::from_value(ckpt.metadata.clone())?;
        let model = config.init(device)?;
        ckpt.restore(&model, true)?;
        Ok(model)
    }

    /// No-grad prediction: one forward, device argmax, aux at the predicted
    /// target. Returns `(target_logits [B,Q,N+1], aux [B,Q,aux_width])` tensors.
    pub fn predict(&self, batch: &PlannerBatch<R, E>) -> Result<(Tensor<R, E>, Tensor<R, E>)> {
        let _guard = crate::autograd::no_grad();
        let tiles = Var::constant(batch.tiles.clone());
        let glob = Var::constant(batch.glob.clone());
        let units = Var::constant(batch.units.clone());
        let out = self.forward(&tiles, &glob, &units, &batch.unit_onehot)?;
        let ids = crate::tensor::ops::reduce::argmax(out.target_logits.tensor(), 2)?;
        let oh = crate::tensor::ops::index::one_hot::<R, E>(&ids, self.config.tiles() + 1)?;
        let aux = self.heads(&out, &oh)?;
        Ok((out.target_logits.into_tensor(), aux.into_tensor()))
    }

    /// No-grad aux outputs at the given (teacher-forced) targets.
    pub fn predict_aux(
        &self,
        batch: &PlannerBatch<R, E>,
        target_onehot: &Tensor<R, E>,
    ) -> Result<Tensor<R, E>> {
        let _guard = crate::autograd::no_grad();
        let tiles = Var::constant(batch.tiles.clone());
        let glob = Var::constant(batch.glob.clone());
        let units = Var::constant(batch.units.clone());
        let out = self.forward(&tiles, &glob, &units, &batch.unit_onehot)?;
        Ok(self.heads(&out, target_onehot)?.into_tensor())
    }
}

impl<R: Runtime, E: FloatElem> Module<R, E> for TaskPlanner<R, E> {
    fn visit(&self, visitor: &mut ModuleVisitor<'_, R, E>) {
        visitor.child("tile_in1", &self.tile_in1);
        visitor.child("tile_in2", &self.tile_in2);
        visitor.child("glob_in1", &self.glob_in1);
        visitor.child("glob_in2", &self.glob_in2);
        visitor.child("unit_in1", &self.unit_in1);
        visitor.child("unit_in2", &self.unit_in2);
        visitor.param("tile_pos", &self.tile_pos);
        visitor.param("step_emb", &self.step_emb);
        visitor.param("none_key", &self.none_key);
        for (i, block) in self.tile_blocks.iter().enumerate() {
            visitor.child_at("tile_blocks", i, block);
        }
        for (i, block) in self.joint_blocks.iter().enumerate() {
            visitor.child_at("joint_blocks", i, block);
        }
        visitor.child("norm", &self.norm);
        visitor.child("q_proj", &self.q_proj);
        visitor.child("k_proj", &self.k_proj);
        visitor.child("aux", &self.aux);
    }
}

/// One turn batch in plain host layout (the npz layout of §1.2).
pub struct HostBatch {
    /// Batch of turns.
    pub turns: usize,
    /// `[B, N, c_tile]` row-major floats.
    pub tiles: Vec<f32>,
    /// `[B, c_glob]` floats.
    pub glob: Vec<f32>,
    /// `[B, U, c_unit]` floats.
    pub units: Vec<f32>,
    /// `[B, U]` tile index per unit, `-1` = padding.
    pub upos: Vec<i32>,
    /// `[B, U, K]` visit target: `0..99` tile, `100` NONE, `-100` ignore.
    pub tgt: Vec<i32>,
    /// `[B, U, K]` first op `0..12`, `-100` ignore.
    pub op: Vec<i32>,
    /// `[B, U, K]` crop `0..4`, `-100` ignore.
    pub crop: Vec<i32>,
    /// `[B, U, K, 13]` multi-hot ops, 0/1.
    pub opset: Vec<u8>,
    /// `[B, U]` turns until visit 1's first op, `-1` = none.
    pub eta: Vec<i32>,
}

/// Step weights for the K visit slots (visit 1 counts full, later visits half).
const STEP_W: [f32; 3] = [1.0, 0.5, 0.5];

/// A device batch with every keep mask and divisor precomputed on the host.
///
/// `target_keep` is pre-multiplied by the step weight; `weights` holds the
/// five divisors (each keep sum clamped to >= 1).
pub struct PlannerBatch<R: Runtime, E: FloatElem> {
    /// `[B, N, c_tile]` tile features.
    pub tiles: Tensor<R, E>,
    /// `[B, c_glob]` global features.
    pub glob: Tensor<R, E>,
    /// `[B, U, c_unit]` unit features.
    pub units: Tensor<R, E>,
    /// `[B, U, N]` unit position one-hot (padding rows are all zero).
    pub unit_onehot: Tensor<R, E>,
    /// `[B*Q]` target ids, 0 where ignored.
    pub target_ids: IdTensor<R>,
    /// `[B*Q]` step weight where kept, else 0.
    pub target_keep: Tensor<R, E>,
    /// `[B, Q, N+1]` one-hot of the true target (teacher forcing), zero row if ignored.
    pub target_onehot: Tensor<R, E>,
    /// `[B*Q]` op ids, 0 where ignored.
    pub op_ids: IdTensor<R>,
    /// `[B*Q]` 1 where the op loss applies.
    pub op_keep: Tensor<R, E>,
    /// `[B*Q]` crop ids, 0 where ignored.
    pub crop_ids: IdTensor<R>,
    /// `[B*Q]` 1 where the crop loss applies.
    pub crop_keep: Tensor<R, E>,
    /// `[B*Q, n_ops]` multi-hot op targets.
    pub opset: Tensor<R, E>,
    /// `[B*Q]` = op_keep.
    pub opset_keep: Tensor<R, E>,
    /// `[B*Q]` log1p(eta) on step 0, else 0.
    pub eta_log: Tensor<R, E>,
    /// `[B*Q]` 1 on step 0 with eta >= 0, else 0.
    pub eta_keep: Tensor<R, E>,
    /// The five divisors: sums of the keep masks, each clamped to >= 1.
    pub weights: [f32; 5],
}

impl<R: Runtime, E: FloatElem> Clone for PlannerBatch<R, E> {
    fn clone(&self) -> Self {
        Self {
            tiles: self.tiles.clone(),
            glob: self.glob.clone(),
            units: self.units.clone(),
            unit_onehot: self.unit_onehot.clone(),
            target_ids: self.target_ids.clone(),
            target_keep: self.target_keep.clone(),
            target_onehot: self.target_onehot.clone(),
            op_ids: self.op_ids.clone(),
            op_keep: self.op_keep.clone(),
            crop_ids: self.crop_ids.clone(),
            crop_keep: self.crop_keep.clone(),
            opset: self.opset.clone(),
            opset_keep: self.opset_keep.clone(),
            eta_log: self.eta_log.clone(),
            eta_keep: self.eta_keep.clone(),
            weights: self.weights,
        }
    }
}

impl<R: Runtime, E: FloatElem> PlannerBatch<R, E> {
    /// Build a device batch from host arrays, computing every mask and
    /// divisor on the host. Queries are unit-major: `q = u*K + j`.
    pub fn from_host(
        cfg: &TaskPlannerConfig,
        h: &HostBatch,
        device: &Device<R>,
    ) -> Result<Self> {
        let (b, n, u, kk, q) = (h.turns, cfg.tiles(), cfg.max_units, cfg.k, cfg.queries());
        let (ct, cg, cu, no, nc) = (cfg.c_tile, cfg.c_glob, cfg.c_unit, cfg.n_ops, cfg.n_crops);
        let expect = |name: &str, got: usize, want: usize| -> Result<()> {
            if got != want {
                return Err(Error::shape(format!(
                    "planner batch {name} holds {got} elements, expected {want}"
                )));
            }
            Ok(())
        };
        expect("tiles", h.tiles.len(), b * n * ct)?;
        expect("glob", h.glob.len(), b * cg)?;
        expect("units", h.units.len(), b * u * cu)?;
        expect("upos", h.upos.len(), b * u)?;
        expect("tgt", h.tgt.len(), b * q)?;
        expect("op", h.op.len(), b * q)?;
        expect("crop", h.crop.len(), b * q)?;
        expect("opset", h.opset.len(), b * q * no)?;
        expect("eta", h.eta.len(), b * u)?;

        let mut unit_oh = vec![0.0f32; b * u * n];
        for (i, &p) in h.upos.iter().enumerate() {
            if p == -1 {
                continue;
            }
            if !(0 <= p && (p as usize) < n) {
                return Err(Error::shape(format!("upos[{i}] = {p} is not -1 or a tile 0..{n}")));
            }
            unit_oh[i * n + p as usize] = 1.0;
        }

        let mut target_ids = vec![0u32; b * q];
        let mut target_keep = vec![0.0f32; b * q];
        let mut target_oh = vec![0.0f32; b * q * (n + 1)];
        let mut op_ids = vec![0u32; b * q];
        let mut op_keep = vec![0.0f32; b * q];
        let mut crop_ids = vec![0u32; b * q];
        let mut crop_keep = vec![0.0f32; b * q];
        let mut opset = vec![0.0f32; b * q * no];
        let mut eta_log = vec![0.0f32; b * q];
        let mut eta_keep = vec![0.0f32; b * q];

        for bi in 0..b {
            for uu in 0..u {
                let step0_q = (bi * u + uu) * kk;
                let eta = h.eta[bi * u + uu];
                if eta != -1 && eta < 0 {
                    return Err(Error::shape(format!(
                        "eta[{}] = {eta} is not -1 or non-negative",
                        bi * u + uu
                    )));
                }
                for j in 0..kk {
                    let qi = step0_q + j;
                    let flat = bi * q + uu * kk + j;
                    // Target.
                    let t = h.tgt[flat];
                    if t == -100 {
                        // Ignored: id 0, keep 0, zero one-hot row.
                    } else if (0 <= t && (t as usize) < n) || t as usize == n {
                        target_ids[flat] = t as u32;
                        let w = if kk == 3 { STEP_W[j] } else { 1.0 };
                        target_keep[flat] = w;
                        target_oh[flat * (n + 1) + t as usize] = 1.0;
                    } else {
                        return Err(Error::shape(format!(
                            "tgt[{qi}] = {t} is not -100, 0..{n} or 100 (NONE)"
                        )));
                    }
                    let tgt_is_tile = t >= 0 && (t as usize) < n;
                    // First op: kept where the target is a real tile and op is labelled.
                    let o = h.op[flat];
                    if o != -100 && !(0 <= o && (o as usize) < no) {
                        return Err(Error::shape(format!(
                            "op[{qi}] = {o} is not -100 or 0..{no}"
                        )));
                    }
                    if tgt_is_tile && o >= 0 {
                        op_ids[flat] = o as u32;
                        op_keep[flat] = 1.0;
                    }
                    // Crop: kept where labelled.
                    let c = h.crop[flat];
                    if c != -100 && !(0 <= c && (c as usize) < nc) {
                        return Err(Error::shape(format!(
                            "crop[{qi}] = {c} is not -100 or 0..{nc}"
                        )));
                    }
                    if c >= 0 {
                        crop_ids[flat] = c as u32;
                        crop_keep[flat] = 1.0;
                    }
                    // Opset targets.
                    for i in 0..no {
                        let v = h.opset[flat * no + i];
                        if v > 1 {
                            return Err(Error::shape(format!(
                                "opset[{qi},{i}] = {v} is not 0 or 1"
                            )));
                        }
                        opset[flat * no + i] = v as f32;
                    }
                    // Eta: step 0 only, where eta >= 0.
                    if j == 0 && eta >= 0 {
                        eta_log[flat] = ((eta as f32) + 1.0).ln();
                        eta_keep[flat] = 1.0;
                    }
                }
            }
        }

        let sum = |xs: &[f32]| xs.iter().sum::<f32>().max(1.0);
        let weights = [
            sum(&target_keep),
            sum(&op_keep),
            sum(&crop_keep),
            sum(&op_keep),
            sum(&eta_keep),
        ];
        Ok(Self {
            tiles: Tensor::from_f32(&h.tiles, vec![b, n, ct], device)?,
            glob: Tensor::from_f32(&h.glob, vec![b, cg], device)?,
            units: Tensor::from_f32(&h.units, vec![b, u, cu], device)?,
            unit_onehot: Tensor::from_f32(&unit_oh, vec![b, u, n], device)?,
            target_ids: IdTensor::from_slice(&target_ids, vec![b * q], device)?,
            target_keep: Tensor::from_f32(&target_keep, vec![b * q], device)?,
            target_onehot: Tensor::from_f32(&target_oh, vec![b, q, n + 1], device)?,
            op_ids: IdTensor::from_slice(&op_ids, vec![b * q], device)?,
            op_keep: Tensor::from_f32(&op_keep, vec![b * q], device)?,
            crop_ids: IdTensor::from_slice(&crop_ids, vec![b * q], device)?,
            crop_keep: Tensor::from_f32(&crop_keep, vec![b * q], device)?,
            opset: Tensor::from_f32(&opset, vec![b * q, no], device)?,
            opset_keep: Tensor::from_f32(&op_keep, vec![b * q], device)?,
            eta_log: Tensor::from_f32(&eta_log, vec![b * q], device)?,
            eta_keep: Tensor::from_f32(&eta_keep, vec![b * q], device)?,
            weights,
        })
    }
}

/// Imitation-learning task over [`PlannerBatch`]es.
pub struct PlannerTask<'a, R: Runtime, E: FloatElem> {
    model: &'a TaskPlanner<R, E>,
    params: Vec<Param<R, E>>,
    loss_scale: f32,
}

impl<'a, R: Runtime, E: FloatElem> PlannerTask<'a, R, E> {
    /// Train every parameter of `model` with `loss_scale` 1.
    pub fn new(model: &'a TaskPlanner<R, E>) -> Self {
        Self {
            params: model.parameters(),
            model,
            loss_scale: 1.0,
        }
    }

    /// Static loss scale (§2.5): multiplies the loss; scale `eps` and the
    /// trainer clip by the same factor so the update is unchanged.
    pub fn with_loss_scale(mut self, scale: f32) -> Self {
        self.loss_scale = scale;
        self
    }

    /// The five unscaled component losses `[target, op, opset, crop, eta]`
    /// (before the 1 / 0.3 / 0.1 weights and the loss scale).
    pub fn component_losses(&self, b: &PlannerBatch<R, E>) -> Result<[Var<R, E>; 5]> {
        let cfg = self.model.config();
        let (n, no, nc, aw) = (cfg.tiles(), cfg.n_ops, cfg.n_crops, cfg.aux_width());
        let bq = b.target_ids.len();
        let out = self.model.forward(
            &Var::constant(b.tiles.clone()),
            &Var::constant(b.glob.clone()),
            &Var::constant(b.units.clone()),
            &b.unit_onehot,
        )?;
        // Target pointer loss, normalised by the step-weighted keep sum.
        let ce_t = out
            .target_logits
            .reshape(vec![bq, n + 1])?
            .cross_entropy_rows(&b.target_ids, 0.0)?;
        let l_t = ce_t
            .mul(&Var::constant(b.target_keep.clone()))?
            .sum()?
            .mul_scalar(1.0 / b.weights[0]);
        // Aux heads at the true target.
        let aux = self.model.heads(&out, &b.target_onehot)?.reshape(vec![bq, aw])?;
        let parts = aux.split(&[no, no, nc, 1], 1)?;
        let (op, os, cr, eta) = (&parts[0], &parts[1], &parts[2], &parts[3]);
        let l_op = op
            .cross_entropy_rows(&b.op_ids, 0.0)?
            .mul(&Var::constant(b.op_keep.clone()))?
            .sum()?
            .mul_scalar(1.0 / b.weights[1]);
        let l_cr = cr
            .cross_entropy_rows(&b.crop_ids, 0.0)?
            .mul(&Var::constant(b.crop_keep.clone()))?
            .sum()?
            .mul_scalar(1.0 / b.weights[2]);
        // BCE with logits, elementwise: softplus(x) - x*y.
        let bce = os
            .softplus()?
            .sub(&os.mul(&Var::constant(b.opset.clone()))?)?;
        let l_os = bce
            .sum_dim(1)?
            .reshape(vec![bq])?
            .mul(&Var::constant(b.opset_keep.clone()))?
            .sum()?
            .mul_scalar(1.0 / (b.weights[3] * no as f32));
        // Eta MSE on log1p.
        let diff = eta
            .reshape(vec![bq])?
            .sub(&Var::constant(b.eta_log.clone()))?;
        let l_eta = diff
            .mul(&diff)?
            .mul(&Var::constant(b.eta_keep.clone()))?
            .sum()?
            .mul_scalar(1.0 / b.weights[4]);
        Ok([l_t, l_op, l_os, l_cr, l_eta])
    }
}

impl<R: Runtime, E: FloatElem> crate::train::trainer::TrainStep<R, E> for PlannerTask<'_, R, E> {
    type Batch = PlannerBatch<R, E>;

    fn parameters(&self) -> Vec<Param<R, E>> {
        self.params.clone()
    }

    fn loss(&self, b: &Self::Batch) -> Result<Var<R, E>> {
        let [l_t, l_op, l_os, l_cr, l_eta] = self.component_losses(b)?;
        l_t
            .add(&l_op)?
            .add(&l_os.mul_scalar(0.3))?
            .add(&l_cr.mul_scalar(0.3))?
            .add(&l_eta.mul_scalar(0.1))
            .map(|v| v.mul_scalar(self.loss_scale))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_cfg() -> TaskPlannerConfig {
        let mut c = TaskPlannerConfig::default();
        c.grid = 2;
        c.max_units = 2;
        c.k = 2;
        c.d_model = 8;
        c.n_tile_layers = 1;
        c.n_joint_layers = 1;
        c.ssm.n_heads = 2;
        c.ssm.n_groups = 2;
        c.ssm.head_dim = 4;
        c.ssm.d_state = 4;
        c.ssm.chunk_size = 4;
        c.seed = 0;
        c
    }

    #[test]
    fn transpose_grid_twice_is_identity_and_swaps_axes() {
        use crate::backends::Auto;
        type R = Auto;
        let device = Device::<R>::default();
        // 2x2 grid, d = 1: values [a, b, c, d] row-major = rows [a b] / [c d].
        let x = Var::constant(Tensor::<R, f32>::from_f32(&[1.0, 2.0, 3.0, 4.0], vec![1, 4, 1], &device).unwrap());
        let t = transpose_grid(&x, 2).unwrap();
        // (y, x) -> (x, y): row-major [a c b d].
        assert_eq!(t.tensor().to_f32(), vec![1.0, 3.0, 2.0, 4.0]);
        let tt = transpose_grid(&t, 2).unwrap();
        assert_eq!(tt.tensor().to_f32(), x.tensor().to_f32());
    }

    #[test]
    fn biblock_output_shape_finite_and_bidirectional() {
        use crate::backends::Auto;
        type R = Auto;
        let device = Device::<R>::default();
        let cfg = tiny_cfg();
        let mut rng = Rng::seeded(0);
        let block = BiBlock::<R, f32>::new(&cfg, 2, &device, &mut rng).unwrap();
        // [1, 4, 8]: tile tokens (grid 2 -> 4 tiles).
        let data: Vec<f32> = (0..32).map(|i| (i as f32) * 0.01).collect();
        let x = Var::constant(Tensor::<R, f32>::from_f32(&data, vec![1, 4, 8], &device).unwrap());
        let y = block.apply(&x).unwrap();
        assert_eq!(y.shape().dims(), &[1, 4, 8]);
        assert!(y.tensor().to_f32().iter().all(|v| v.is_finite()));
        // Bidirectionality: changing only the LAST token must change token 0.
        let mut data2 = data.clone();
        data2[3 * 8] += 1.0;
        let x2 = Var::constant(Tensor::<R, f32>::from_f32(&data2, vec![1, 4, 8], &device).unwrap());
        let y2 = block.apply(&x2).unwrap();
        let (a, b) = (y.tensor().to_f32(), y2.tensor().to_f32());
        let delta: f32 = a.iter().take(8).zip(b.iter().take(8)).map(|(p, q)| (p - q).abs()).sum();
        assert!(delta > 1e-6, "token 0 ignores the last token: not bidirectional (delta={delta})");
    }
}
