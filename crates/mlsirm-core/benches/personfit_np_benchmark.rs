use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn personfit_np_benchmark(c: &mut Criterion) {
    let n = 100;
    let pi: Vec<f64> = (0..n).map(|i| (i as f64) / (n as f64)).collect();
    let lo: Vec<f64> = (0..n).map(|i| (i as f64)).collect();

    let mut group = c.benchmark_group("personfit_np_moments");

    group.bench_function("multiple_iters", |b| {
        b.iter(|| {
            let s1: f64 = pi.iter().zip(&lo).map(|(&p, &l)| p * l).sum();
            let s2: f64 = pi.iter().zip(&lo).map(|(&p, &l)| p * (1.0 - p) * l).sum();
            let s3: f64 = pi.iter().sum();
            let s4: f64 = pi.iter().map(|&p| p * (1.0 - p)).sum();
            let beta: f64 = pi.iter().zip(&lo).map(|(&p, &l)| p * (1.0 - p) * l * l).sum::<f64>() - s2 * s2 / s4;
            black_box((s1, s2, s3, s4, beta));
        })
    });

    group.bench_function("single_fold", |b| {
        b.iter(|| {
            let (s1, s2, s3, s4, beta_term) = pi.iter().zip(&lo).fold(
                (0.0, 0.0, 0.0, 0.0, 0.0),
                |(a1, a2, a3, a4, ab), (&p, &l)| {
                    let p_comp = 1.0 - p;
                    let p_p_comp = p * p_comp;
                    (
                        a1 + p * l,
                        a2 + p_p_comp * l,
                        a3 + p,
                        a4 + p_p_comp,
                        ab + p_p_comp * l * l,
                    )
                }
            );
            let beta = beta_term - s2 * s2 / s4;
            black_box((s1, s2, s3, s4, beta));
        })
    });

    group.finish();
}

criterion_group!(benches, personfit_np_benchmark);
criterion_main!(benches);
