use mlsirm_core::samefit_grm::{import_summary, PackedArchive};
#[test]
fn imports_hash_bound_synthetic_summary_and_rejects_tampering() {
    use sha2::{Digest, Sha256};
    let request = r#"{"y":[0,1,2,0],"observed":null,"group_id":[0,1],"n_groups":2,"n_cat":3,"item_ids":["i1","i2"],"q_general":121}"#;
    let archive = PackedArchive::new(
        vec![true, true],
        2,
        3,
        vec![
            0.0,
            0.6931471805599453,
            0.32663425997828094,
            0.0,
            0.6931471805599453,
            0.32663425997828094,
            0.0,
            0.0,
        ],
    )
    .unwrap();
    let summary = serde_json::json!({"anchor":[true,true],"coordinate_map":archive.coordinates().unwrap(),"packed":archive.packed_bits().iter().map(|&b|f64::from_bits(b)).collect::<Vec<_>>(),"q_general":121,"loglik_cpu_f64":-4.0*3.0f64.ln(),"fit_identity":"synthetic"});
    let hash = format!("{:x}", Sha256::digest(request.as_bytes()));
    let fit = import_summary(request, &hash, &summary, 1e-10).unwrap();
    assert!((fit.loglik() + 4.0 * 3.0f64.ln()).abs() < 1e-10);
    assert!(import_summary(request, "bad", &summary, 1e-10).is_err());
    let mut altered = summary.clone();
    altered["coordinate_map"][0] = serde_json::json!({"Mean":1});
    assert!(import_summary(request, &hash, &altered, 1e-10).is_err());
    altered = summary.clone();
    altered["loglik_cpu_f64"] = 0.into();
    assert!(import_summary(request, &hash, &altered, 1e-10).is_err());
}
