use super::*;

#[test]
fn local_growth_student_model_wires_phase_controller() {
    let config = NpaConfig::growing_3dgs();
    let model =
        local_growth_student_model(config.clone(), 19, 0.0, LOCAL_GROWTH_EXPANSION_GAIN).unwrap();
    let input_dims = config.perception_dims();
    let output_dims = config.update_dims();
    let phase_channel = growth_3d_phase_channel(config.state_dims).unwrap();
    let phase_output = config.spatial_dims + phase_channel;

    let mut features = vec![0.0_f32; 2 * input_dims];
    features[GROWTH_3D_LIVENESS_CHANNEL] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    features[config.state_dims + GROWTH_3D_LIVENESS_CHANNEL] = 0.0;
    let far_base = input_dims;
    features[far_base + GROWTH_3D_LIVENESS_CHANNEL] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    features[far_base + config.state_dims + GROWTH_3D_LIVENESS_CHANNEL] =
        GROWTH_3D_INACTIVE_OPACITY_LOGIT;

    let update = model.forward_update_from_features(&features).unwrap();

    assert!(
        update[phase_output] > 0.0,
        "near-front local liveness contrast should advance the phase state"
    );
    assert!(
        update[phase_output] > update[output_dims + phase_output].abs() + 1.0,
        "phase controller should dominate the random initialized far-row phase baseline"
    );
}
#[test]
fn local_growth_student_model_uses_phase_for_material_maturation() {
    let config = NpaConfig::growing_3dgs();
    let model =
        local_growth_student_model(config.clone(), 29, 0.0, LOCAL_GROWTH_EXPANSION_GAIN).unwrap();
    let input_dims = config.perception_dims();
    let output_dims = config.update_dims();
    let phase_channel = growth_3d_phase_channel(config.state_dims).unwrap();
    let material_channel = growth_3d_material_opacity_channel(config.state_dims).unwrap();
    let material_output = config.spatial_dims + material_channel;

    let mut features = vec![0.0_f32; 2 * input_dims];
    features[GROWTH_3D_LIVENESS_CHANNEL] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    features[phase_channel] = 0.0;
    let mature_base = input_dims;
    features[mature_base + GROWTH_3D_LIVENESS_CHANNEL] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    features[mature_base + phase_channel] = 0.75;

    let update = model.forward_update_from_features(&features).unwrap();
    let immature_material = update[material_output];
    let mature_material = update[output_dims + material_output];

    assert!(
        mature_material > immature_material + 0.15,
        "mature local phase should produce stronger material opacity growth"
    );
}
#[test]
fn local_growth_student_model_uses_phase_to_boost_local_front_liveness() {
    let config = NpaConfig::growing_3dgs();
    let model =
        local_growth_student_model(config.clone(), 31, 0.0, LOCAL_GROWTH_EXPANSION_GAIN).unwrap();
    let input_dims = config.perception_dims();
    let output_dims = config.update_dims();
    let phase_channel = growth_3d_phase_channel(config.state_dims).unwrap();
    let liveness_output = config.spatial_dims + GROWTH_3D_LIVENESS_CHANNEL;
    let phase_liveness_hidden = 25;

    assert_eq!(
        model.weights.b1[phase_liveness_hidden], -4.0,
        "phase liveness bridge should require local-front contrast before it activates"
    );
    assert_eq!(
        model.weights.w1
            [phase_liveness_hidden * input_dims + config.state_dims + GROWTH_3D_LIVENESS_CHANNEL],
        1.0
    );
    assert_eq!(
        model.weights.w1[phase_liveness_hidden * input_dims + GROWTH_3D_LIVENESS_CHANNEL],
        -1.0
    );
    assert_eq!(
        model.weights.w1[phase_liveness_hidden * input_dims + phase_channel],
        1.0
    );
    assert_eq!(
        model.weights.w2[liveness_output * config.hidden_dims + phase_liveness_hidden],
        LOCAL_GROWTH_PHASE_LIVENESS_GAIN
    );

    let mut features = vec![0.0_f32; 3 * input_dims];
    features[GROWTH_3D_LIVENESS_CHANNEL] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    features[config.state_dims + GROWTH_3D_LIVENESS_CHANNEL] = 0.0;
    features[phase_channel] = 0.0;
    let phased_front_base = input_dims;
    features[phased_front_base + GROWTH_3D_LIVENESS_CHANNEL] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    features[phased_front_base + config.state_dims + GROWTH_3D_LIVENESS_CHANNEL] = 0.0;
    features[phased_front_base + phase_channel] = 0.75;
    let far_base = 2 * input_dims;
    features[far_base + GROWTH_3D_LIVENESS_CHANNEL] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    features[far_base + config.state_dims + GROWTH_3D_LIVENESS_CHANNEL] =
        GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    features[far_base + phase_channel] = 0.75;

    let update = model.forward_update_from_features(&features).unwrap();
    let unphased_front_liveness = update[liveness_output];
    let phased_front_liveness = update[output_dims + liveness_output];
    let far_liveness = update[2 * output_dims + liveness_output];

    assert!(
        phased_front_liveness > unphased_front_liveness + 0.02,
        "phase memory should make a local-front dormant row easier to activate"
    );
    assert!(
        far_liveness < unphased_front_liveness * 0.25,
        "phase without local-front liveness contrast must not globally activate dormant rows"
    );
}
#[test]
fn local_growth_student_model_materializes_active_rows_without_waking_dormant_material() {
    let config = NpaConfig::growing_3dgs();
    let model =
        local_growth_student_model(config.clone(), 33, 0.0, LOCAL_GROWTH_EXPANSION_GAIN).unwrap();
    let input_dims = config.perception_dims();
    let output_dims = config.update_dims();
    let material_channel = growth_3d_material_opacity_channel(config.state_dims).unwrap();
    let material_output = config.spatial_dims + material_channel;
    let active_material_hidden = 26;

    assert_eq!(model.weights.b1[active_material_hidden], 1.0);
    assert_eq!(
        model.weights.w1[active_material_hidden * input_dims + GROWTH_3D_LIVENESS_CHANNEL],
        1.0
    );
    assert_eq!(
        model.weights.w1[active_material_hidden * input_dims + material_channel],
        -0.25
    );
    assert_eq!(
        model.weights.w2[material_output * config.hidden_dims + active_material_hidden],
        LOCAL_GROWTH_ACTIVE_MATERIAL_GAIN
    );

    let mut features = vec![0.0_f32; 4 * input_dims];
    features[GROWTH_3D_LIVENESS_CHANNEL] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    features[material_channel] = GROWTH_3D_MATERIAL_INACTIVE_OPACITY_LOGIT;
    let active_low_base = input_dims;
    features[active_low_base + GROWTH_3D_LIVENESS_CHANNEL] = 0.0;
    features[active_low_base + material_channel] = GROWTH_3D_MATERIAL_INACTIVE_OPACITY_LOGIT;
    let active_mid_base = 2 * input_dims;
    features[active_mid_base + GROWTH_3D_LIVENESS_CHANNEL] = 0.0;
    features[active_mid_base + material_channel] = 0.0;
    let active_high_base = 3 * input_dims;
    features[active_high_base + GROWTH_3D_LIVENESS_CHANNEL] = 0.0;
    features[active_high_base + material_channel] = GROWTH_3D_VISIBLE_MATERIAL_OPACITY_TARGET;

    let update = model.forward_update_from_features(&features).unwrap();
    let dormant_material = update[material_output];
    let active_low_material = update[output_dims + material_output];
    let active_mid_material = update[2 * output_dims + material_output];
    let active_high_material = update[3 * output_dims + material_output];

    assert!(
        active_low_material > dormant_material + 0.5,
        "newly active low-material rows should receive a strong materialization update"
    );
    assert!(
        active_mid_material > dormant_material + 0.2,
        "already live material rows should keep materializing until visible"
    );
    assert!(
        active_high_material < active_mid_material,
        "materialization bridge should damp itself once material opacity is high"
    );
    assert!(
        dormant_material < active_low_material * 0.25,
        "dormant rows must not become material-visible without liveness"
    );
}
#[test]
fn local_growth_student_model_sustains_active_liveness_without_global_activation() {
    let config = NpaConfig::growing_3dgs();
    let model =
        local_growth_student_model(config.clone(), 37, 0.0, LOCAL_GROWTH_EXPANSION_GAIN).unwrap();
    let input_dims = config.perception_dims();
    let output_dims = config.update_dims();
    let liveness_output = config.spatial_dims + GROWTH_3D_LIVENESS_CHANNEL;
    let active_liveness_low_hidden = 17;
    let active_liveness_high_hidden = 18;

    assert_eq!(
        model.weights.b1[active_liveness_low_hidden], 1.0,
        "active liveness low hidden should gate on liveness + 1"
    );
    assert_eq!(
        model.weights.w1[active_liveness_low_hidden * input_dims + GROWTH_3D_LIVENESS_CHANNEL],
        1.0
    );
    assert_eq!(
        model.weights.w1[active_liveness_high_hidden * input_dims + GROWTH_3D_LIVENESS_CHANNEL],
        1.0
    );
    assert_eq!(
        model.weights.w2[liveness_output * config.hidden_dims + active_liveness_low_hidden],
        LOCAL_GROWTH_ACTIVE_LIVENESS_GAIN
    );
    assert_eq!(
        model.weights.w2[liveness_output * config.hidden_dims + active_liveness_high_hidden],
        -2.0 * LOCAL_GROWTH_ACTIVE_LIVENESS_GAIN
    );

    let mut features = vec![0.0_f32; 3 * input_dims];
    features[GROWTH_3D_LIVENESS_CHANNEL] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    features[config.state_dims + GROWTH_3D_LIVENESS_CHANNEL] = GROWTH_3D_INACTIVE_OPACITY_LOGIT;
    let active_base = input_dims;
    features[active_base + GROWTH_3D_LIVENESS_CHANNEL] = 0.0;
    features[active_base + config.state_dims + GROWTH_3D_LIVENESS_CHANNEL] = 0.0;
    let saturated_base = 2 * input_dims;
    features[saturated_base + GROWTH_3D_LIVENESS_CHANNEL] = 2.0;
    features[saturated_base + config.state_dims + GROWTH_3D_LIVENESS_CHANNEL] = 2.0;

    let update = model.forward_update_from_features(&features).unwrap();
    let dormant_update = update[liveness_output];
    let active_update = update[output_dims + liveness_output];
    let saturated_update = update[2 * output_dims + liveness_output];

    assert!(
        active_update > dormant_update + LOCAL_GROWTH_ACTIVE_LIVENESS_GAIN * 0.5,
        "active liveness should be sustained without globally activating dormant substrate rows"
    );
    assert!(
        saturated_update < active_update,
        "bounded active liveness controller should push back once liveness is already high"
    );
}
#[test]
fn local_growth_student_model_wires_velocity_memory_to_motion_and_damping() {
    let config = NpaConfig::growing_3dgs();
    let model =
        local_growth_student_model(config.clone(), 41, 0.0, LOCAL_GROWTH_EXPANSION_GAIN).unwrap();
    let input_dims = config.perception_dims();
    let output_dims = config.update_dims();
    let velocity_channels = growth_3d_velocity_channels(config.state_dims).unwrap();
    let velocity_channels = velocity_channels.collect::<Vec<_>>();

    for (axis, &velocity_channel) in velocity_channels.iter().enumerate() {
        let pos_hidden = 19 + axis * 2;
        let neg_hidden = pos_hidden + 1;
        assert_eq!(
            model.weights.w1[pos_hidden * input_dims + velocity_channel],
            1.0
        );
        assert_eq!(
            model.weights.w1[neg_hidden * input_dims + velocity_channel],
            -1.0
        );
        assert_eq!(
            model.weights.w2[axis * config.hidden_dims + pos_hidden],
            LOCAL_GROWTH_VELOCITY_OUTPUT_GAIN
        );
        assert_eq!(
            model.weights.w2[axis * config.hidden_dims + neg_hidden],
            -LOCAL_GROWTH_VELOCITY_OUTPUT_GAIN
        );
    }

    let mut features = vec![0.0_f32; 2 * input_dims];
    features[velocity_channels[0]] = 0.25;
    let negative_base = input_dims;
    features[negative_base + velocity_channels[1]] = -0.5;
    let update = model.forward_update_from_features(&features).unwrap();

    assert!(
        update[0] > 0.20,
        "positive velocity memory should drive same-axis motion"
    );
    assert!(
        update[config.spatial_dims + velocity_channels[0]]
            < -LOCAL_GROWTH_VELOCITY_DAMPING_GAIN * 0.20,
        "positive velocity memory should decay through its state update"
    );
    assert!(
        update[output_dims + 1] < -0.40,
        "negative velocity memory should drive opposite-axis motion"
    );
    assert!(
        update[output_dims + config.spatial_dims + velocity_channels[1]]
            > LOCAL_GROWTH_VELOCITY_DAMPING_GAIN * 0.40,
        "negative velocity memory should damp back toward zero"
    );
}
