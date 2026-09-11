use mlsirm_core::sampling_design::{
    finite_population_proportion_design, AllocationMethod, SamplingStratum,
};

#[test]
fn maximum_strata_census_allocation_is_bounded_and_exact() {
    let strata = vec![
        SamplingStratum {
            population_size: 1,
            expected_proportion: 0.5,
        };
        100_000
    ];

    let design = finite_population_proportion_design(
        100_000,
        0.95,
        1.0e-6,
        &strata,
        AllocationMethod::Proportional,
    )
    .expect("the maximum admitted census design must terminate and allocate exactly");

    assert_eq!(design.sample_size, 100_000);
    assert_eq!(design.stratum_sample_sizes.len(), 100_000);
    assert!(design.stratum_sample_sizes.iter().all(|count| *count == 1));
    assert_eq!(
        design
            .stratum_sample_sizes
            .iter()
            .copied()
            .sum::<usize>(),
        design.sample_size
    );
}
