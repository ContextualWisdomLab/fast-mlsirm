//! Missing-only consumer of authentic native fits. No study input.
use mlsirm_core::mask_grm::lifecycle::resume;
fn main() {
    if let Err(e) = resume::main_cli() {
        eprintln!("RESUME_REFUSAL {e}");
        std::process::exit(2);
    }
}
