//! G1 tests (ENTITY_MODEL_PLAN.md): JSON round trip of the Kaggriculture
//! spec, one validation error per §1.2 bullet (message names the field), and
//! `chunk_for` values.

use mamba3::models::{
    ContextSetSpec, DecoderMode, EntityModelSpec, HeadSpec, QuerySetSpec, SetLayout,
};

fn kaggriculture_spec() -> EntityModelSpec {
    EntityModelSpec {
        globals: 114,
        context: vec![
            ContextSetSpec::new("tiles", 100, 48).with_layout(SetLayout::Grid {
                height: 10,
                width: 10,
                alternate_axes: true,
            }),
        ],
        queries: Some(
            QuerySetSpec::new("units", 20, 36, 3)
                .with_anchor("tiles")
                .with_autoregressive("target"),
        ),
        heads: vec![
            HeadSpec::pointer("target", "tiles", 1).step_weights(vec![1.0, 0.5, 0.5]),
            HeadSpec::categorical("op", 13).condition_on("target"),
            HeadSpec::multilabel("opset", 13)
                .condition_on("target")
                .loss_weight(0.3),
            HeadSpec::categorical("crop", 5)
                .condition_on("target")
                .loss_weight(0.3),
            HeadSpec::regression("eta", 1)
                .first_step_only()
                .loss_weight(0.1),
        ],
        d_model: 128,
        context_layers: 3,
        decoder_layers: 3,
        decoder: DecoderMode::StepCausal {
            crew_symmetric: true,
        },
        ssm: mamba3::ssm::config::SsmConfig {
            d_model: 128,
            n_heads: 4,
            head_dim: 64,
            d_state: 32,
            n_groups: 1,
            ..Default::default()
        },
        chunk_size: None,
        norm_eps: 1e-5,
        seed: 0,
    }
}

#[test]
fn kaggriculture_spec_json_round_trip_and_valid() {
    let spec = kaggriculture_spec();
    spec.validate().unwrap();
    let json = serde_json::to_string(&spec).unwrap();
    let back: EntityModelSpec = serde_json::from_str(&json).unwrap();
    assert_eq!(spec, back);
    back.validate().unwrap();
    assert_eq!(spec.n_ctx(), 100);
    assert_eq!(spec.set_offset("tiles"), Some(0));
    assert_eq!(spec.query_tokens(), 60);
    assert_eq!(spec.plan_head().unwrap().name, "target");
    assert_eq!(spec.head("op").unwrap().width(&spec), Some(13));
    assert_eq!(spec.head("target").unwrap().width(&spec), Some(101));
}

#[test]
fn chunk_for_values() {
    assert_eq!(EntityModelSpec::chunk_for(100), 50);
    assert_eq!(EntityModelSpec::chunk_for(160), 40);
    assert_eq!(EntityModelSpec::chunk_for(7), 32);
}

fn err_msg(spec: &EntityModelSpec) -> String {
    spec.validate().unwrap_err().to_string()
}

#[test]
fn duplicate_set_names_rejected() {
    let mut spec = kaggriculture_spec();
    spec.context.push(ContextSetSpec::new("tiles", 5, 2));
    let msg = err_msg(&spec);
    assert!(msg.contains("tiles"), "{msg}");
    assert!(msg.contains("duplicate"), "{msg}");
}

#[test]
fn duplicate_head_names_rejected() {
    let mut spec = kaggriculture_spec();
    spec.heads.push(HeadSpec::categorical("op", 4));
    let msg = err_msg(&spec);
    assert!(msg.contains("op"), "{msg}");
    assert!(msg.contains("duplicate"), "{msg}");
}

#[test]
fn pointer_set_naming_nothing_rejected() {
    let mut spec = kaggriculture_spec();
    spec.heads[0] = HeadSpec::pointer("target", "nope", 1).step_weights(vec![1.0, 0.5, 0.5]);
    let msg = err_msg(&spec);
    assert!(msg.contains("set"), "{msg}");
    assert!(msg.contains("nope"), "{msg}");
}

#[test]
fn anchor_naming_nothing_rejected() {
    let mut spec = kaggriculture_spec();
    spec.queries.as_mut().unwrap().anchor = Some("nope".to_string());
    let msg = err_msg(&spec);
    assert!(msg.contains("anchor"), "{msg}");
}

#[test]
fn condition_on_naming_nothing_rejected() {
    let mut spec = kaggriculture_spec();
    spec.heads[1] = HeadSpec::categorical("op", 13).condition_on("nope");
    let msg = err_msg(&spec);
    assert!(msg.contains("condition_on"), "{msg}");
}

#[test]
fn condition_on_non_pointer_rejected() {
    let mut spec = kaggriculture_spec();
    // "op" is categorical, not a pointer.
    spec.heads[2] = HeadSpec::multilabel("opset", 13).condition_on("op");
    let msg = err_msg(&spec);
    assert!(msg.contains("condition_on"), "{msg}");
    assert!(msg.contains("not a pointer"), "{msg}");
}

#[test]
fn condition_on_pointer_head_rejected() {
    // Pointer heads read the query state alone.
    let mut spec = kaggriculture_spec();
    spec.heads[1] = HeadSpec::categorical("op", 13);
    spec.heads[0] = HeadSpec::pointer("target", "tiles", 1)
        .step_weights(vec![1.0, 0.5, 0.5])
        .condition_on("target");
    let msg = err_msg(&spec);
    assert!(msg.contains("condition_on"), "{msg}");
}

#[test]
fn autoregressive_on_naming_nothing_rejected() {
    let mut spec = kaggriculture_spec();
    spec.queries.as_mut().unwrap().autoregressive_on = Some("nope".to_string());
    let msg = err_msg(&spec);
    assert!(msg.contains("autoregressive_on"), "{msg}");
}

#[test]
fn autoregressive_on_non_pointer_rejected() {
    let mut spec = kaggriculture_spec();
    spec.queries.as_mut().unwrap().autoregressive_on = Some("op".to_string());
    let msg = err_msg(&spec);
    assert!(msg.contains("autoregressive_on"), "{msg}");
    assert!(msg.contains("not a pointer"), "{msg}");
}

#[test]
fn grid_mismatch_rejected() {
    let mut spec = kaggriculture_spec();
    spec.context[0].layout = SetLayout::Grid {
        height: 9,
        width: 10,
        alternate_axes: true,
    };
    let msg = err_msg(&spec);
    assert!(msg.contains("layout"), "{msg}");
}

#[test]
fn joint_with_autoregression_rejected() {
    let mut spec = kaggriculture_spec();
    spec.decoder = DecoderMode::Joint;
    let msg = err_msg(&spec);
    assert!(msg.contains("decoder"), "{msg}");
    assert!(msg.contains("autoregressive_on"), "{msg}");
}

#[test]
fn zero_steps_rejected() {
    let mut spec = kaggriculture_spec();
    spec.queries.as_mut().unwrap().steps = 0;
    let msg = err_msg(&spec);
    assert!(msg.contains("steps"), "{msg}");
}

#[test]
fn step_weights_length_mismatch_rejected() {
    let mut spec = kaggriculture_spec();
    spec.heads[0] = HeadSpec::pointer("target", "tiles", 1).step_weights(vec![1.0, 0.5]);
    let msg = err_msg(&spec);
    assert!(msg.contains("step_weights"), "{msg}");
}

#[test]
fn first_on_plan_head_rejected() {
    let mut spec = kaggriculture_spec();
    spec.heads[0] = HeadSpec::pointer("target", "tiles", 1)
        .first_step_only()
        .step_weights(vec![1.0, 0.5, 0.5]);
    let msg = err_msg(&spec);
    assert!(msg.contains("First"), "{msg}");
    assert!(msg.contains("plan head"), "{msg}");
}

#[test]
fn zero_sizes_rejected() {
    let mut spec = kaggriculture_spec();
    spec.d_model = 0;
    let msg = err_msg(&spec);
    assert!(msg.contains("d_model"), "{msg}");

    let mut spec = kaggriculture_spec();
    spec.heads[1] = HeadSpec::categorical("op", 0);
    let msg = err_msg(&spec);
    assert!(msg.contains("classes"), "{msg}");
}

#[test]
fn ssm_validated_with_d_model() {
    let mut spec = kaggriculture_spec();
    spec.ssm.n_heads = 0;
    let msg = err_msg(&spec);
    assert!(msg.contains("ssm"), "{msg}");
}

#[test]
fn from_entity_set() {
    let rl_set = mamba3::rl::spec::EntitySet::new("foes", 7, 4);
    let ctx = ContextSetSpec::from(&rl_set);
    assert_eq!(ctx.name, "foes");
    assert_eq!(ctx.count, 7);
    assert_eq!(ctx.features, 4);
}
