use super::*;
#[test]
fn native_capture_and_packed_values_are_lossless() {
    let (r, c, y, g, a) = complete_fixture();
    let state = fixture_state(&r, &[-1, -1]);
    let params: Vec<Vec<ItemParams>> = state
        .items
        .iter()
        .map(|row| {
            row.iter()
                .map(|p| ItemParams {
                    a_g: p.a_general,
                    a_s: p.a_specific,
                    d: p.threshold.clone(),
                })
                .collect()
        })
        .collect();
    assert_eq!(
        ConsumedStateReceipt::capture(
            &params,
            &state.general_mean,
            &state.general_sd,
            &state.specific_mean,
            &state.specific_sd
        ),
        state
    );
    assert_eq!(
        ItemCoordinateReceipt::from_packed(&[1.25, -0.75, 0.5, -0.2], true),
        ItemCoordinateReceipt {
            a_general: 1.25,
            a_specific: Some(-0.75),
            threshold: vec![0.5, -0.2]
        }
    );
    assert_eq!(
        ItemCoordinateReceipt::from_packed(&[1.25, 0.5, -0.2], false),
        ItemCoordinateReceipt {
            a_general: 1.25,
            a_specific: None,
            threshold: vec![0.5, -0.2]
        }
    );
    assert!(valid(&r, &c, &y, &g, &a));
}

// These synthetic states intentionally differ by sweep and by start. Only
// coordinate transport is tested: no optimization/E-step is called.
fn varying_fixture() -> (
    BifactorMultigroupResult,
    BifactorMultigroupConfig,
    Vec<usize>,
    Vec<usize>,
    Vec<bool>,
) {
    let (mut r, c, y, g, a) = complete_fixture();
    for sweep in &mut r.dispatch {
        let state = sweep.input.as_mut().unwrap();
        if sweep.start == 0 {
            state.general_mean[1] = 0.7;
            state.items[1][1].threshold[0] = 0.8;
        }
        if sweep.start == 1 && sweep.iteration < 2 {
            state.general_mean[1] = if sweep.iteration == 0 { 0.0 } else { 0.05 };
            state.items[1][1].threshold[0] = if sweep.iteration == 0 { 0.2 } else { 0.3 };
        }
    }
    for idx in [0, 2, 3] {
        let input = r.dispatch[idx].input.clone().unwrap();
        let output = r.dispatch[idx + 1].input.clone().unwrap();
        let sweep = &mut r.dispatch[idx];
        for row in &mut sweep.items {
            row.input = Some(input.items[row.group_scope.unwrap_or(0)][row.item].clone());
            row.output = Some(output.items[row.group_scope.unwrap_or(0)][row.item].clone());
        }
        let group = sweep.group_update.as_mut().unwrap();
        let mut cm = input;
        cm.items = output.items.clone();
        group.input = Some(cm);
        group.coordinates[0].value = output.general_mean[1];
    }
    reseal(&mut r);
    (r, c, y, g, a)
}
#[test]
fn different_sweep_coordinates_and_nonwinning_bank_are_valid() {
    let (r, c, y, g, a) = varying_fixture();
    assert!(valid(&r, &c, &y, &g, &a));
    assert_ne!(r.dispatch[0].input, r.dispatch[2].input);
    assert_ne!(r.dispatch[2].input, r.dispatch[3].input);
    assert_ne!(r.dispatch[3].input, r.dispatch[4].input);
}
macro_rules! value_mutation {
    ($name:ident,$r:ident,$body:block) => {
        #[test]
        fn $name() {
            let (mut $r, c, y, g, a) = varying_fixture();
            assert!(valid(&$r, &c, &y, &g, &a));
            $body;
            reseal(&mut $r);
            assert!(!valid(&$r, &c, &y, &g, &a));
        }
    };
}
value_mutation!(nonwinning_item_input_contradiction, r, {
    r.dispatch[0].items[2].input.as_mut().unwrap().a_general = 42.0;
});
value_mutation!(winning_pooled_item_output_contradiction, r, {
    r.dispatch[3].items[0].output.as_mut().unwrap().threshold[0] = 42.0;
});
value_mutation!(free_focal_output_contradiction, r, {
    r.dispatch[2].items[2].output.as_mut().unwrap().a_general = 42.0;
});
value_mutation!(group_cm_consumes_updated_item_bank, r, {
    r.dispatch[3]
        .group_update
        .as_mut()
        .unwrap()
        .input
        .as_mut()
        .unwrap()
        .items[1][1]
        .threshold[0] = 42.0;
});
value_mutation!(group_cm_consumes_same_sweep_old_prior, r, {
    r.dispatch[3]
        .group_update
        .as_mut()
        .unwrap()
        .input
        .as_mut()
        .unwrap()
        .general_mean[1] = 0.1;
});
value_mutation!(
    terminal_winner_and_final_cannot_contradict_returned_mean,
    r,
    {
        r.dispatch[3].group_update.as_mut().unwrap().coordinates[0].value = 42.0;
        r.dispatch[4].input.as_mut().unwrap().general_mean[1] = 42.0;
        r.execution_evidence.as_mut().unwrap().final_posterior[0]
            .input
            .as_mut()
            .unwrap()
            .general_mean[1] = 42.0;
    }
);
value_mutation!(terminal_winner_cannot_contradict_returned_item, r, {
    r.dispatch[3].items[2].output.as_mut().unwrap().threshold[0] = 42.0;
    r.dispatch[3]
        .group_update
        .as_mut()
        .unwrap()
        .input
        .as_mut()
        .unwrap()
        .items[1][1]
        .threshold[0] = 42.0;
    r.dispatch[4].input.as_mut().unwrap().items[1][1].threshold[0] = 42.0;
    r.execution_evidence.as_mut().unwrap().final_posterior[0]
        .input
        .as_mut()
        .unwrap()
        .items[1][1]
        .threshold[0] = 42.0;
});
value_mutation!(final_posterior_must_consume_winning_terminal_bank, r, {
    r.execution_evidence.as_mut().unwrap().final_posterior[0]
        .input
        .as_mut()
        .unwrap()
        .general_mean[1] = 42.0;
});
value_mutation!(fixed_coordinate_cannot_change_between_sweeps, r, {
    r.dispatch[3].input.as_mut().unwrap().general_sd[0] = 1.2;
});
value_mutation!(missing_sweep_consumption_is_incomplete, r, {
    r.dispatch[2].input = None;
});
value_mutation!(missing_item_consumption_is_incomplete, r, {
    r.dispatch[3].items[1].input = None;
});
value_mutation!(missing_item_return_is_incomplete, r, {
    r.dispatch[3].items[1].output = None;
});
value_mutation!(missing_group_consumption_is_incomplete, r, {
    r.dispatch[2].group_update.as_mut().unwrap().input = None;
});
value_mutation!(missing_final_consumption_is_incomplete, r, {
    r.execution_evidence.as_mut().unwrap().final_posterior[0].input = None;
});
#[test]
fn dispatch_payload_is_in_result_association() {
    let (mut r, c, y, g, a) = varying_fixture();
    assert!(valid(&r, &c, &y, &g, &a));
    let before = r.evidence_result_binding().unwrap();
    // Still a valid raw counter relationship; stale association must reject.
    r.dispatch[0].items[0].stages.line_search += 1;
    assert_ne!(before, r.evidence_result_binding().unwrap());
    assert!(!valid(&r, &c, &y, &g, &a));
    reseal(&mut r);
    assert!(valid(&r, &c, &y, &g, &a));
}
#[test]
fn legacy_schema_remains_readable_but_incomplete() {
    let (mut r, c, y, g, a) = complete_fixture();
    r.execution_evidence.as_mut().unwrap().schema_version = 1;
    let saved = serde_json::to_vec(&r).unwrap();
    r = serde_json::from_slice(&saved).unwrap();
    reseal(&mut r);
    assert!(!valid(&r, &c, &y, &g, &a));
}
#[test]
fn general_reflection_uses_native_anchored_orientation() {
    let (mut r, c, y, g, a) = complete_fixture();
    for sweep in &mut r.dispatch {
        let mut state = sweep.input.take().unwrap();
        for row in &mut state.items {
            for item in row {
                item.a_general = -item.a_general;
            }
        }
        state.general_mean[1] = -state.general_mean[1];
        sweep.input = Some(state.clone());
        for item in &mut sweep.items {
            let p = state.items[item.group_scope.unwrap_or(0)][item.item].clone();
            item.input = Some(p.clone());
            item.output = Some(p);
        }
        if let Some(group) = sweep.group_update.as_mut() {
            group.input = Some(state);
            group.coordinates[0].value = -group.coordinates[0].value;
        }
    }
    r.execution_evidence.as_mut().unwrap().final_posterior[0].input = r.dispatch[4].input.clone();
    reseal(&mut r);
    assert!(valid(&r, &c, &y, &g, &a));
    r.general_mean[1] = -r.general_mean[1];
    reseal(&mut r);
    assert!(!valid(&r, &c, &y, &g, &a));
}
#[test]
fn free_general_outlier_cannot_choose_reflection() {
    let (mut r, c, y, g, a) = complete_fixture();
    r.a_general[1][1] = -100.0;
    bind_fixture_coordinates(&mut r, &[-1, -1]);
    reseal(&mut r);
    assert!(valid(&r, &c, &y, &g, &a));
}
fn specific_fixture() -> (
    BifactorMultigroupResult,
    BifactorMultigroupConfig,
    Vec<usize>,
    Vec<usize>,
    Vec<bool>,
) {
    let (mut r, mut c, y, g, a) = complete_fixture();
    c.estimate_specific_means = true;
    c.estimate_specific_vars = true;
    r.a_specific = vec![vec![0.7, 0.8]; 2];
    r.specific_mean = vec![vec![0.0], vec![0.15]];
    r.specific_sd = vec![vec![1.0], vec![1.2]];
    bind_fixture_coordinates(&mut r, &[0, 0]);
    for sweep in &mut r.dispatch {
        if let Some(group) = sweep.group_update.as_mut() {
            group.coordinates.push(GroupCoordinateReceipt {
                group: 1,
                specific: Some(0),
                is_sd: false,
                owner: "gpu".into(),
                value: 0.15,
            });
            group.coordinates.push(GroupCoordinateReceipt {
                group: 1,
                specific: Some(0),
                is_sd: true,
                owner: "gpu".into(),
                value: 1.2,
            });
        }
    }
    let e = r.execution_evidence.as_mut().unwrap();
    e.final_posterior[0].n_specific = 1;
    e.request_sha256 = receipts::request_binding(&y, None, &g, 2, &[0, 0], &a, &c).unwrap();
    reseal(&mut r);
    (r, c, y, g, a)
}
#[test]
fn specific_mean_sd_and_reflection_are_reconciled() {
    let (mut r, c, y, g, a) = specific_fixture();
    assert!(r
        .validate_gpu_evidence(&y, None, &g, 2, &[0, 0], &a, &c, false)
        .is_ok());
    for sweep in &mut r.dispatch {
        let mut state = sweep.input.take().unwrap();
        for row in &mut state.items {
            for p in row {
                p.a_specific = p.a_specific.map(|x| -x);
            }
        }
        state.specific_mean[1][0] = -state.specific_mean[1][0];
        sweep.input = Some(state.clone());
        for item in &mut sweep.items {
            let p = state.items[item.group_scope.unwrap_or(0)][item.item].clone();
            item.input = Some(p.clone());
            item.output = Some(p);
        }
        if let Some(group) = sweep.group_update.as_mut() {
            group.input = Some(state);
            group.coordinates[2].value = -group.coordinates[2].value;
        }
    }
    r.execution_evidence.as_mut().unwrap().final_posterior[0].input = r.dispatch[4].input.clone();
    reseal(&mut r);
    assert!(r
        .validate_gpu_evidence(&y, None, &g, 2, &[0, 0], &a, &c, false)
        .is_ok());
    let mut wrong = r.clone();
    wrong.specific_mean[1][0] = -wrong.specific_mean[1][0];
    reseal(&mut wrong);
    assert!(wrong
        .validate_gpu_evidence(&y, None, &g, 2, &[0, 0], &a, &c, false)
        .is_err());
    r.dispatch[3].group_update.as_mut().unwrap().coordinates[3].value = 42.0;
    reseal(&mut r);
    assert!(r
        .validate_gpu_evidence(&y, None, &g, 2, &[0, 0], &a, &c, false)
        .is_err());
}
#[test]
fn zero_iteration_nonwinning_start_needs_only_consumed_terminal() {
    let (mut r, c, y, g, a) = complete_fixture();
    r.start_outcomes[0] = serde_json::json!({"start":0,"embedded":false,"converged":false,"termination_reason":"numerical_em_stall","n_iter":0,"loglik_trace":[-3.0]});
    r.dispatch.remove(0);
    r.dispatch[0].iteration = 0;
    r.m_step_gpu_dispatches -= 3;
    reseal(&mut r);
    assert!(valid(&r, &c, &y, &g, &a));
}

fn reseal(r: &mut BifactorMultigroupResult) {
    let binding = r.evidence_result_binding().unwrap();
    if let Some(e) = r.execution_evidence.as_mut() {
        e.result_sha256 = binding;
    }
}
// Synthetic-only repeated coordinates. No native fit or statistical arithmetic.
fn fixture_state(r: &BifactorMultigroupResult, map: &[i32]) -> ConsumedStateReceipt {
    let ni = map.len();
    let m1 = r.threshold[0].len() / ni;
    ConsumedStateReceipt {
        items: (0..r.a_general.len())
            .map(|g| {
                (0..ni)
                    .map(|i| ItemCoordinateReceipt {
                        a_general: r.a_general[g][i],
                        a_specific: (map[i] >= 0).then(|| r.a_specific[g][i]),
                        threshold: r.threshold[g][i * m1..(i + 1) * m1].to_vec(),
                    })
                    .collect()
            })
            .collect(),
        general_mean: r.general_mean.clone(),
        general_sd: r.general_sd.clone(),
        specific_mean: r.specific_mean.clone(),
        specific_sd: r.specific_sd.clone(),
    }
}
fn bind_fixture_coordinates(r: &mut BifactorMultigroupResult, map: &[i32]) {
    let state = fixture_state(r, map);
    for sweep in &mut r.dispatch {
        sweep.input = Some(state.clone());
        for item in &mut sweep.items {
            item.input = Some(state.items[item.group_scope.unwrap_or(0)][item.item].clone());
            item.output = item.input.clone();
        }
        if let Some(group) = sweep.group_update.as_mut() {
            group.input = Some(state.clone());
        }
    }
    if let Some(e) = r.execution_evidence.as_mut() {
        for f in &mut e.final_posterior {
            f.input = Some(state.clone());
        }
    }
}
#[test]
fn terminal_group_coordinate_matches_returned_winning_bank() {
    let (mut r, c, y, g, a) = complete_fixture();
    assert!(valid(&r, &c, &y, &g, &a));
    r.dispatch[3].group_update.as_mut().unwrap().coordinates[0].value = 42.0;
    reseal(&mut r);
    assert!(!valid(&r, &c, &y, &g, &a));
}
#[test]
fn f4_item_output_must_match_next_consumed_bank() {
    let (mut r, c, y, g, a) = complete_fixture();
    assert!(valid(&r, &c, &y, &g, &a));
    let mut row = serde_json::to_value(&r.dispatch[3].items[0]).unwrap();
    row["output"] = serde_json::json!({"a_general":42.0,"a_specific":null,"threshold":[0.2]});
    r.dispatch[3].items[0] = serde_json::from_value(row).unwrap();
    reseal(&mut r);
    assert!(
        !valid(&r, &c, &y, &g, &a),
        "item output not reconciled to the following E-step bank"
    );
}
#[test]
fn f4_group_output_must_match_nonwinning_next_sweep() {
    let (mut r, c, y, g, a) = complete_fixture();
    assert!(valid(&r, &c, &y, &g, &a));
    r.dispatch[0].group_update.as_mut().unwrap().coordinates[0].value = 42.0;
    reseal(&mut r);
    assert!(
        !valid(&r, &c, &y, &g, &a),
        "nonwinning group output not reconciled to next sweep"
    );
}
#[test]
fn f4_sweep_input_coordinates_must_be_consumed() {
    let (mut r, c, y, g, a) = complete_fixture();
    assert!(valid(&r, &c, &y, &g, &a));
    let mut row = serde_json::to_value(&r.dispatch[4]).unwrap();
    row["input"]["general_mean"] = serde_json::json!([0.0, 42.0]);
    r.dispatch[4] = serde_json::from_value(row).unwrap();
    reseal(&mut r);
    assert!(!valid(&r, &c, &y, &g, &a), "terminal E-step input ignored");
}

fn fixture() -> (
    BifactorMultigroupResult,
    BifactorMultigroupConfig,
    Vec<usize>,
    Vec<usize>,
    Vec<bool>,
) {
    let cfg = BifactorMultigroupConfig {
        q_general: 7,
        q_specific: 1,
        max_iter: 4,
        tol: 1e-8,
        n_starts: 2,
        seed: 42,
        newton_iter: 20,
        ridge: 1e-6,
        slope_prior: SlopePrior::None,
        estimate_specific_vars: false,
        estimate_specific_means: false,
        device: crate::Device::Gpu,
    };
    let y = vec![0, 1, 1, 0];
    let groups = vec![0, 1];
    let anchor = vec![true, false];
    let r:BifactorMultigroupResult=serde_json::from_value(serde_json::json!({
  "scoring_posterior_gpu":true,"start_outcomes":[{"start":0,"embedded":false,"converged":true,"termination_reason":"tolerance_met","n_iter":1,"loglik_trace":[-3.0,-2.5]},{"start":1,"embedded":false,"converged":true,"termination_reason":"tolerance_met","n_iter":2,"loglik_trace":[-3.0,-2.7,-2.4]}],
  "dispatch":[{"start":0,"iteration":0,"gpu_used":true,"backend":"Metal","device_name":"synthetic","fallback_reason":null}],"dispatch_complete":true,"m_step_gpu_dispatches":9,
  "a_general":[[1.0,1.0],[1.0,1.1]],"a_specific":[[0.0,0.0],[0.0,0.0]],"threshold":[[0.2,0.3],[0.2,0.4]],"general_mean":[0.0,0.1],"general_sd":[1.0,1.1],"specific_sd":[[],[]],"specific_mean":[[],[]],"theta_g_eap":[0.0,0.1],"theta_g_sd":[0.5,0.6],"group_category_counts":[[1,0,0,1],[0,1,1,0]],"loglik_trace":[-3.0,-2.7,-2.4],"n_iter":2,"converged":true,"termination_reason":"tolerance_met","final_loglik_change":0.3,"best_start":1,"n_parameters":8,"slope_prior":"None","em_objective_trace":[-3.0,-2.7,-2.4]
 })).unwrap();
    (r, cfg, y, groups, anchor)
}
#[test]
fn missing_interior_sweep_is_not_dispatch_complete() {
    let (r, cfg, y, g, a) = fixture();
    assert!(
        r.validate_gpu_evidence(&y, None, &g, 2, &[-1, -1], &a, &cfg, false)
            .is_err(),
        "missing all-start/sweep evidence must be rejected"
    );
}
fn complete_fixture() -> (
    BifactorMultigroupResult,
    BifactorMultigroupConfig,
    Vec<usize>,
    Vec<usize>,
    Vec<bool>,
) {
    let (mut r, cfg, y, g, a) = fixture();
    r.dispatch.clear();
    for (start, k) in [(0, 1), (1, 2)] {
        for iteration in 0..=k {
            let mut items = Vec::new();
            let mut group_update = None;
            if iteration < k {
                for (item, scope) in [(0, None), (1, Some(0)), (1, Some(1))] {
                    items.push(ItemStageReceipt {
                        start,
                        iteration,
                        item,
                        group_scope: scope,
                        input: None,
                        output: None,
                        stages: NativeStageReceipt {
                            objective: 7,
                            gradient: 4,
                            hessian: 4,
                            prior: 7,
                            solve: 3,
                            line_search: 5,
                            parameter_update: 3,
                            m_step_gpu_dispatches: 1,
                            ..Default::default()
                        },
                    });
                }
                group_update = Some(GroupStageReceipt {
                    start,
                    iteration,
                    completed: true,
                    iteration_limited: false,
                    algorithm: "finite_grid_conditional_em".into(),
                    backend: "fixture".into(),
                    device_name: "fixture".into(),
                    input: None,
                    coordinates: vec![
                        GroupCoordinateReceipt {
                            group: 1,
                            specific: None,
                            is_sd: false,
                            owner: "gpu".into(),
                            value: 0.1,
                        },
                        GroupCoordinateReceipt {
                            group: 1,
                            specific: None,
                            is_sd: true,
                            owner: "gpu".into(),
                            value: 1.1,
                        },
                    ],
                });
            }
            r.dispatch.push(ConcurrentSweepReceipt {
                input: None,
                start,
                iteration,
                gpu_used: true,
                backend: Some("fixture".into()),
                device_name: Some("fixture".into()),
                fallback_reason: None,
                items,
                group_update,
            });
        }
    }
    r.execution_evidence = Some(ExecutionEvidence {
        schema_version: 2,
        device: "Gpu".into(),
        configured_starts: 2,
        embedded: false,
        request_sha256: receipts::request_binding(&y, None, &g, 2, &[-1, -1], &a, &cfg).unwrap(),
        result_sha256: String::new(),
        final_posterior: vec![FinalPosteriorReceipt {
            start: 1,
            iteration: 2,
            n_persons: 2,
            n_items: 2,
            n_specific: 0,
            q_general: 7,
            q_specific: 1,
            gpu_used: true,
            backend: Some("fixture".into()),
            device_name: Some("fixture".into()),
            fallback_reason: None,
            output_consumed: true,
            input: None,
        }],
        host_probability_tables: true,
        host_eap: true,
        host_information: true,
        full_gpu: false,
    });
    bind_fixture_coordinates(&mut r, &[-1, -1]);
    reseal(&mut r);
    (r, cfg, y, g, a)
}
fn valid(
    r: &BifactorMultigroupResult,
    cfg: &BifactorMultigroupConfig,
    y: &[usize],
    g: &[usize],
    a: &[bool],
) -> bool {
    r.validate_gpu_evidence(y, None, g, 2, &[-1, -1], a, cfg, false)
        .is_ok()
}
#[test]
fn complete_multistart_mixed_receipts_roundtrip() {
    let (r, c, y, g, a) = complete_fixture();
    assert!(valid(&r, &c, &y, &g, &a));
    let restored: BifactorMultigroupResult =
        serde_json::from_slice(&serde_json::to_vec(&r).unwrap()).unwrap();
    assert!(valid(&restored, &c, &y, &g, &a));
    assert_eq!(
        serde_json::to_value(&r.dispatch).unwrap(),
        serde_json::to_value(&restored.dispatch).unwrap()
    );
}
macro_rules! mutation {
    ($name:ident,$r:ident,$body:block) => {
        #[test]
        fn $name() {
            let (mut $r, c, y, g, a) = complete_fixture();
            $body;
            reseal(&mut $r);
            assert!(!valid(&$r, &c, &y, &g, &a));
        }
    };
}
mutation!(drop_start, r, {
    r.start_outcomes.remove(0);
});
mutation!(duplicate_start, r, {
    r.start_outcomes.push(r.start_outcomes[0].clone());
});
mutation!(embedded_flag, r, {
    r.start_outcomes[1]["embedded"] = serde_json::json!(true);
});
mutation!(missing_first_sweep, r, {
    r.dispatch.remove(0);
});
mutation!(missing_interior_sweep, r, {
    r.dispatch.remove(3);
});
mutation!(missing_terminal_sweep, r, {
    r.dispatch.pop();
});
mutation!(duplicate_sweep, r, {
    r.dispatch[1] = r.dispatch[0].clone();
});
mutation!(reorder_sweep, r, {
    r.dispatch.swap(0, 1);
});
mutation!(wrong_start_sweep, r, {
    r.dispatch[2].start = 0;
});
mutation!(extra_sweep, r, {
    r.dispatch.push(r.dispatch[0].clone());
});
mutation!(equal_count_wrong_item, r, {
    r.dispatch[0].items[0].item = 1;
});
mutation!(missing_pooled_item, r, {
    r.dispatch[0].items.remove(0);
});
mutation!(missing_reference_item, r, {
    r.dispatch[0].items.remove(1);
});
mutation!(missing_focal_item, r, {
    r.dispatch[0].items.pop();
});
mutation!(pooled_is_not_group0, r, {
    r.dispatch[0].items[0].group_scope = Some(0);
});
mutation!(duplicate_item, r, {
    r.dispatch[0].items[2] = r.dispatch[0].items[1].clone();
});
mutation!(wrong_item_start, r, {
    r.dispatch[0].items[0].start = 1;
});
mutation!(post_terminal_item, r, {
    let item = r.dispatch[0].items[0].clone();
    r.dispatch[1].items.push(item);
});
mutation!(post_terminal_group, r, {
    r.dispatch[1].group_update = r.dispatch[0].group_update.clone();
});
mutation!(cpu_fallback, r, {
    r.dispatch[0].items[0].stages.cpu_numerical_fallbacks = 1;
});
mutation!(missing_group, r, {
    r.dispatch[0].group_update = None;
});
mutation!(swapped_group_coordinates, r, {
    r.dispatch[0]
        .group_update
        .as_mut()
        .unwrap()
        .coordinates
        .swap(0, 1);
});
mutation!(fixed_reference_as_free, r, {
    r.dispatch[0].group_update.as_mut().unwrap().coordinates[0].group = 0;
});
mutation!(group_not_consumed, r, {
    r.dispatch[0].group_update.as_mut().unwrap().completed = false;
});
mutation!(missing_final, r, {
    r.execution_evidence
        .as_mut()
        .unwrap()
        .final_posterior
        .clear();
});
mutation!(duplicate_final, r, {
    let e = r.execution_evidence.as_mut().unwrap();
    e.final_posterior.push(e.final_posterior[0].clone());
});
mutation!(wrong_final_start, r, {
    r.execution_evidence.as_mut().unwrap().final_posterior[0].start = 0;
});
mutation!(wrong_final_geometry, r, {
    r.execution_evidence.as_mut().unwrap().final_posterior[0].q_general = 21;
});
mutation!(wrong_request_binding, r, {
    r.execution_evidence.as_mut().unwrap().request_sha256 = "another request".into();
});
mutation!(wrong_result_binding, r, {
    r.a_general[0][0] = 1.2;
});
mutation!(legacy_absent_evidence, r, {
    r.execution_evidence = None;
});
mutation!(fake_complete_empty_rows, r, {
    r.dispatch.clear();
});
mutation!(limited_terminal_not_convergence, r, {
    r.dispatch[3].items[0].stages.iteration_limit_updates = 1;
});
#[test]
fn stationary_zero_update_is_valid() {
    let (mut r, c, y, g, a) = complete_fixture();
    r.dispatch[0].items[0].stages = NativeStageReceipt {
        objective: 1,
        gradient: 1,
        hessian: 1,
        prior: 1,
        ..Default::default()
    };
    reseal(&mut r);
    assert!(valid(&r, &c, &y, &g, &a));
}
#[test]
fn bounded_early_update_is_retained() {
    let (mut r, c, y, g, a) = complete_fixture();
    r.dispatch[2].items[0].stages.iteration_limit_updates = 1;
    reseal(&mut r);
    assert!(valid(&r, &c, &y, &g, &a));
}
#[cfg(all(feature = "gpu", not(coverage)))]
#[test]
fn full_native_stage_transport_is_lossless() {
    let original = crate::gpu_mstep::StageReceipt {
        objective: 11,
        gradient: 7,
        hessian: 6,
        prior: 13,
        solve: 5,
        line_search: 17,
        parameter_update: 3,
        e_step_gpu_dispatches: 19,
        m_step_gpu_dispatches: 23,
        cpu_numerical_fallbacks: 29,
        iteration_limit_updates: 31,
    };
    let before = serde_json::to_value(&original).unwrap();
    let neutral: NativeStageReceipt = original.into();
    assert_eq!(before, serde_json::to_value(neutral).unwrap());
}

#[test]
fn appended_embedded_start_has_exact_membership() {
    let (mut r, c, y, g, a) = complete_fixture();
    r.start_outcomes.push(serde_json::json!({"start":2,"embedded":true,"converged":true,"termination_reason":"tolerance_met","n_iter":1,"loglik_trace":[-3.0,-2.6]}));
    let mut extra = r.dispatch[..2].to_vec();
    for s in &mut extra {
        s.start = 2;
        for i in &mut s.items {
            i.start = 2;
        }
        if let Some(group) = s.group_update.as_mut() {
            group.start = 2;
        }
    }
    r.dispatch.extend(extra);
    r.m_step_gpu_dispatches += 3;
    r.execution_evidence.as_mut().unwrap().embedded = true;
    reseal(&mut r);
    assert!(r
        .validate_gpu_evidence(&y, None, &g, 2, &[-1, -1], &a, &c, true)
        .is_ok());
    let mut missing = r.clone();
    missing.start_outcomes.pop();
    assert!(missing
        .validate_gpu_evidence(&y, None, &g, 2, &[-1, -1], &a, &c, true)
        .is_err());
    assert!(r
        .validate_gpu_evidence(&y, None, &g, 2, &[-1, -1], &a, &c, false)
        .is_err());
}
#[test]
fn mean_only_ownership_is_explicit_and_never_full_gpu() {
    let (mut r, mut c, y, g, a) = complete_fixture();
    c.estimate_specific_means = true;
    let map = [0, 0];
    for sweep in r.dispatch.iter_mut().filter(|s| s.group_update.is_some()) {
        let group = sweep.group_update.as_mut().unwrap();
        group.algorithm = "posterior_normal_moments".into();
        group.coordinates.push(GroupCoordinateReceipt {
            group: 1,
            specific: Some(0),
            is_sd: false,
            owner: "cpu_f64".into(),
            value: 0.15,
        });
    }
    r.a_specific = vec![vec![0.7, 0.7]; 2];
    r.specific_mean = vec![vec![0.0], vec![0.15]];
    r.specific_sd = vec![vec![1.0]; 2];
    bind_fixture_coordinates(&mut r, &map);
    let e = r.execution_evidence.as_mut().unwrap();
    e.request_sha256 = receipts::request_binding(&y, None, &g, 2, &map, &a, &c).unwrap();
    e.final_posterior[0].n_specific = 1;
    reseal(&mut r);
    assert!(r
        .validate_gpu_evidence(&y, None, &g, 2, &map, &a, &c, false)
        .is_ok());
    assert!(!r.execution_evidence.as_ref().unwrap().full_gpu);
    r.dispatch[0].group_update.as_mut().unwrap().coordinates[2].owner = "gpu".into();
    reseal(&mut r);
    assert!(r
        .validate_gpu_evidence(&y, None, &g, 2, &map, &a, &c, false)
        .is_err());
}
mutation!(contradictory_group_backend, r, {
    r.dispatch[0].group_update.as_mut().unwrap().backend.clear();
});
mutation!(wrong_coordinate_role, r, {
    r.dispatch[0].group_update.as_mut().unwrap().coordinates[0].specific = Some(0);
});
mutation!(post_terminal_group_limit, r, {
    r.dispatch[3]
        .group_update
        .as_mut()
        .unwrap()
        .iteration_limited = true;
});
