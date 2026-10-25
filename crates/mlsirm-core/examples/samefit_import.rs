//! Bound saved-result consumer: no refit, no participant rows in output.
use mlsirm_core::samefit_grm::{decode, import_summary};
use sha2::{Digest, Sha256};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 6 {
        return Err("request result result_sha256 fit_identity output".into());
    }
    let request = std::fs::read_to_string(&args[1])?;
    let raw = std::fs::read(&args[2])?;
    if format!("{:x}", Sha256::digest(&raw)) != args[3] {
        return Err("result hash mismatch".into());
    }
    let root: serde_json::Value = serde_json::from_slice(&raw)?;
    let mut matches = vec![];
    for round in root["result"]["rounds"].as_array().ok_or("rounds")? {
        for item in round["item_results"].as_array().ok_or("items")? {
            for role in ["null", "alternative"] {
                let v = &item[role];
                if v["fit_identity"].as_str() == Some(&args[4]) {
                    matches.push(v);
                }
            }
        }
    }
    let summary = matches.first().ok_or("identity not found")?;
    for other in &matches {
        if other["packed"] != summary["packed"]
            || other["coordinate_map"] != summary["coordinate_map"]
            || other["anchor"] != summary["anchor"]
            || other["q_general"] != summary["q_general"]
            || other["loglik_cpu_f64"] != summary["loglik_cpu_f64"]
        {
            return Err("identity collision".into());
        }
    }
    let fit = import_summary(
        &request,
        root["request_sha256"].as_str().ok_or("request hash")?,
        summary,
        1e-8,
    )?;
    let d = decode(fit.archive())?;
    let mut output = serde_json::json!({"schema_version":1,"historical_identity":fit.historical_identity,"historical_identity_recomputed":false,"request_sha256":root["request_sha256"],"result_sha256":args[3],"source_core_sha256":root["core_source_sha256"],"source_fitter_sha256":root["fitter_source_sha256"],"item_ids":fit.item_ids(),"anchor":summary["anchor"],"q_general":fit.quadpts(),"coordinate_map":fit.archive().coordinates()?,"packed_archive":serde_json::from_str::<serde_json::Value>(&fit.archive().to_json()?)?,"decoded":d,"native_terminal_loglik":fit.loglik(),"saved_terminal_loglik":summary["loglik_cpu_f64"],"ll_atol":1e-8,"samefit_scope":"packed/map/input/terminal_LL; historical dense bit equality unproven","optimization_calls":0});
    #[cfg(feature = "gpu")]
    if std::env::var("FMLS_SAMEFIT_DIAGNOSTICS").as_deref() == Ok("1") {
        output["diagnostics"] = mlsirm_core::samefit_grm_joint::joint_diagnostics(
            fit.responses(),
            fit.groups(),
            fit.archive(),
            fit.quadpts(),
        )?;
    }
    let mut f = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args[5])?;
    use std::io::Write;
    f.write_all(serde_json::to_string_pretty(&output)?.as_bytes())?;
    println!(
        "IMPORTED {} repeated_locations={} optimization_calls=0",
        args[4],
        matches.len()
    );
    Ok(())
}
