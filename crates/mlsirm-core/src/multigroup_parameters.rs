// Borrowed parameter transport recovered from the native mask producer.
pub(crate) struct MultigroupParameters<'a> {
    pub a_general: &'a [Vec<f64>], pub a_specific: &'a [Vec<f64>],
    pub threshold: &'a [Vec<f64>], pub general_mean: &'a [f64],
    pub general_sd: &'a [f64], pub specific_mean: &'a [Vec<f64>],
    pub specific_sd: &'a [Vec<f64>],
}
