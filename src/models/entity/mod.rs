//! Domain-free entity-to-plan model (ENTITY_MODEL_PLAN.md).
//!
//! G1 lands the spec types; G2–G5 add the encoder, decoder, batch, loss and
//! predict pieces in this module's siblings. Nothing here names a domain.

pub mod batch;
pub mod blocks;
pub mod loss;
pub mod model;
pub mod rl;
pub mod spec;

pub use batch::{
    DatasetHead, DatasetHeadKind, DatasetLayout, EntityBatch, EntityDataset, HeadLabels, HostArrays,
};
pub use blocks::{BiBlock, DecoderLayer, ForwardBlock, Permutation, transpose_grid};
pub use loss::{EntityTask, LossComponents};
pub use rl::{
    Acted, ActedHost, EntityActorCritic, EntityPpoBatch, EntityPpoStats, EntityPpoTask,
    EntityValueHead, actions_to_arrays, check_rl_support, entity_gae, entity_ppo_objective,
};
pub use model::{
    ChoiceIds, CoreLogits, CtxEncoder, Decode, DecoderOut, EntityMetrics, EntityModel, HeadInput,
    HeadOutputs, HeadRun, Prediction, PtrHead, QueryEncoder, StemVars, fused_entity_model,
    set_fused_entity_model,
};
pub use spec::{
    ContextSetSpec, DecoderMode, EntityModelSpec, HeadKind, HeadSpec, QuerySetSpec, SetLayout,
    StepSelection,
};
