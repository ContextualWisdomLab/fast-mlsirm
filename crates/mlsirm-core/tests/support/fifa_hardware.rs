use std::process::Command;

pub fn assert_hardware_cases_are_ignored(expected: &[&str]) {
    let output = Command::new(std::env::current_exe().expect("current native test executable"))
        .args(["--list", "--ignored"])
        .output()
        .expect("list native ignored test registrations");
    assert!(output.status.success(), "native test listing must succeed");
    let listing = String::from_utf8(output.stdout).expect("UTF-8 test names");
    let mut actual: Vec<_> = listing
        .lines()
        .filter_map(|line| line.strip_suffix(": test"))
        .collect();
    let mut expected = expected.to_vec();
    actual.sort_unstable();
    expected.sort_unstable();
    assert_eq!(
        actual, expected,
        "physical GPU cases must require explicit selection, while validation stays in normal CI"
    );
}
