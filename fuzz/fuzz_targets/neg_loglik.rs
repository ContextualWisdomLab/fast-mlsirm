#![no_main]

use libfuzzer_sys::fuzz_target;
use mlsirm_core::{neg_loglik_and_grad, ModelConfig, ModelType, Params, PenaltyConfig};

fn byte_to_f64(byte: u8) -> f64 {
    (byte as f64 / 32.0) - 4.0
}

fn next_value(data: &[u8], cursor: &mut usize) -> f64 {
    let value = byte_to_f64(data[*cursor % data.len()]);
    *cursor += 1;
    value
}

fuzz_target!(|data: &[u8]| {
    if data.len() < 8 {
        return;
    }

    let n_persons = (data[0] as usize % 3) + 1;
    let n_items = (data[1] as usize % 3) + 1;
    let n_dims = (data[2] as usize % 2) + 1;
    let latent_dim = (data[3] as usize % 2) + 1;
    let model_type = match data[4] % 5 {
        0 => ModelType::Mirt,
        1 => ModelType::Mls2plm,
        2 => ModelType::Mlsrm,
        3 => ModelType::Uls2plm,
        _ => ModelType::Ulsrm,
    };
    let eps_distance = 1e-8 + (data[5] as f64 / 255.0) * 1e-3;
    let config = ModelConfig {
        n_persons,
        n_items,
        n_dims,
        latent_dim,
        model_type,
        eps_distance,
    };

    let mut cursor = 6;

    let y = (0..n_persons * n_items)
        .map(|_| {
            if next_value(data, &mut cursor) >= 0.0 {
                1.0
            } else {
                0.0
            }
        })
        .collect::<Vec<_>>();
    let mask = (0..n_persons * n_items)
        .map(|_| next_value(data, &mut cursor) >= -3.5)
        .collect::<Vec<_>>();
    let factor_id = (0..n_items)
        .map(|idx| (idx + data[(cursor + idx) % data.len()] as usize) % n_dims)
        .collect::<Vec<_>>();
    cursor += n_items;

    let params = Params {
        theta: (0..n_persons * n_dims)
            .map(|_| next_value(data, &mut cursor))
            .collect(),
        alpha: (0..n_items)
            .map(|_| next_value(data, &mut cursor))
            .collect(),
        b: (0..n_items)
            .map(|_| next_value(data, &mut cursor))
            .collect(),
        xi: (0..n_persons * latent_dim)
            .map(|_| next_value(data, &mut cursor))
            .collect(),
        zeta: (0..n_items * latent_dim)
            .map(|_| next_value(data, &mut cursor))
            .collect(),
        tau: next_value(data, &mut cursor),
    };
    let penalty = PenaltyConfig {
        lambda_theta: next_value(data, &mut cursor).abs() * 0.01,
        lambda_xi: next_value(data, &mut cursor).abs() * 0.01,
        lambda_zeta: next_value(data, &mut cursor).abs() * 0.01,
        lambda_b: next_value(data, &mut cursor).abs() * 0.01,
        lambda_alpha: next_value(data, &mut cursor).abs() * 0.01,
        lambda_tau: next_value(data, &mut cursor).abs() * 0.01,
        mu_alpha: next_value(data, &mut cursor),
        mu_tau: next_value(data, &mut cursor),
    };

    let _ = neg_loglik_and_grad(&y, Some(&mask), &factor_id, &params, &config, &penalty);
});
