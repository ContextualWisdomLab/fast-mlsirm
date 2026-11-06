use mlsirm_core::samefit_grm::{decode, Coordinate, PackedArchive};

fn fixture() -> PackedArchive {
    PackedArchive::new(
        vec![true, false],
        3,
        3,
        vec![
            1.0, 0.5, 0.0, -0.7, 0.8, 0.0, -0.5, 0.6, 0.0, -0.3, 0.4, 0.0, 0.2, 0.1, -0.2, -0.1,
        ],
    )
    .unwrap()
}

#[test]
fn decode_common_free_and_bits_roundtrip_without_fit() {
    let a = fixture();
    let b = PackedArchive::from_json(&a.to_json().unwrap()).unwrap();
    assert_eq!(a.packed_bits(), b.packed_bits());
    let d = decode(&b).unwrap();
    assert_eq!(d.slope, vec![1.0, -0.7, 1.0, -0.5, 1.0, -0.3]);
    assert_eq!(&d.intercepts[..4], &[0.5, 0.5 - 1.0, 0.8, 0.8 - 1.0]);
    assert_eq!(d.means, vec![0.0, 0.2, -0.2]);
    assert_eq!(d.sds[0], 1.0);
    assert_eq!(a.coordinates().unwrap().last(), Some(&Coordinate::LogSd(2)));
}

#[test]
fn rejects_nonfinite_underflow_schema_and_coordinate_changes() {
    let a = fixture();
    let mut v: serde_json::Value = serde_json::from_str(&a.to_json().unwrap()).unwrap();
    v["schema_version"] = 2.into();
    assert!(PackedArchive::from_json(&v.to_string()).is_err());
    let mut x = vec![1.0, 0.5, -1000.0, 0.0, 0.0];
    assert!(PackedArchive::new(vec![true], 2, 3, x.clone()).is_err());
    x[2] = f64::NAN;
    assert!(PackedArchive::new(vec![true], 2, 3, x).is_err());
    let mut v: serde_json::Value = serde_json::from_str(&a.to_json().unwrap()).unwrap();
    v["anchor"][0] = false.into();
    assert!(PackedArchive::from_json(&v.to_string()).is_err());
}

#[cfg(feature = "gpu")]
#[test]
fn actual_gpu_moments_xi_match_f64_quadrature() {
    use mlsirm_core::samefit_grm_gpu::moments_gpu;
    let a = fixture();
    let d = decode(&a).unwrap();
    let (nodes, weights) = mlsirm_core::samefit_grm::quadrature(121).unwrap();
    let gpu = moments_gpu(&a, &nodes, &weights).expect("requires real GPU; no skip/fallback");
    assert_eq!(gpu.dispatches, 1);
    assert_eq!(gpu.moments.len(), 3 * 8);
    assert_eq!(gpu.xi.len(), 3 * 8 * 8);
    for g in 0..3 {
        let mut means = [0.0; 8];
        let mut cross = [[0.0; 8]; 8];
        // Independent exact category enumeration conditional on each node.
        for (&t, &w) in nodes.iter().zip(&weights) {
            let theta = d.means[g] + d.sds[g] * t;
            let mut p = [[0.0; 3]; 2];
            for i in 0..2 {
                let c1 = 1.0
                    / (1.0 + (-(d.slope[g * 2 + i] * theta + d.intercepts[g * 4 + i * 2])).exp());
                let c2 = 1.0
                    / (1.0
                        + (-(d.slope[g * 2 + i] * theta + d.intercepts[g * 4 + i * 2 + 1])).exp());
                p[i] = [1.0 - c1, c1 - c2, c2];
            }
            for x in 0..3 {
                for y in 0..3 {
                    let z = [
                        (x >= 1) as u8 as f64,
                        (x >= 2) as u8 as f64,
                        (y >= 1) as u8 as f64,
                        (y >= 2) as u8 as f64,
                        (x >= 1 && y >= 1) as u8 as f64,
                        (x >= 1 && y >= 2) as u8 as f64,
                        (x >= 2 && y >= 1) as u8 as f64,
                        (x >= 2 && y >= 2) as u8 as f64,
                    ];
                    let mass = w * p[0][x] * p[1][y];
                    for i in 0..8 {
                        means[i] += mass * z[i];
                        for j in 0..8 {
                            cross[i][j] += mass * z[i] * z[j];
                        }
                    }
                }
            }
        }
        for i in 0..8 {
            assert!((gpu.moments[g * 8 + i] - means[i]).abs() < 2e-6);
            for j in 0..8 {
                assert!(
                    (gpu.xi[g * 64 + i * 8 + j] - (cross[i][j] - means[i] * means[j])).abs() < 3e-6
                );
            }
        }
    }
    println!(
        "GPU_DISPATCH backend={} device={} dispatches={} moments={} xi={}",
        gpu.backend,
        gpu.device_name,
        gpu.dispatches,
        gpu.moments.len(),
        gpu.xi.len()
    );
}
