use serde_json::Value;
use std::io::Write;
use std::process::{Command, Stdio};
use tempfile::tempdir;

#[test]
fn cli_compress_decompress_and_stats_use_same_ccr_db() {
    let tempdir = tempdir().unwrap();
    let db_path = tempdir.path().join("ccr.sqlite");

    let marker = run_hr_with_stdin(&db_path, &["compress", "--input", "-"], "cli test payload");
    let marker = marker.trim();
    assert!(marker.starts_with("<<ccr:"));
    assert!(marker.ends_with(">>"));

    let hash = marker
        .trim()
        .strip_prefix("<<ccr:")
        .and_then(|value| value.strip_suffix(">>"))
        .unwrap();
    assert_eq!(hash.len(), 24);

    let direct = run_hr(&db_path, &["decompress", "--hash", hash]);
    assert_eq!(direct.trim(), "cli test payload");

    let expanded = run_hr_with_stdin(
        &db_path,
        &["decompress", "--input", "-"],
        &format!("before {marker} after"),
    );
    assert_eq!(expanded.trim(), "before cli test payload after");

    let stats = run_hr(&db_path, &["stats"]);
    let stats: Value = serde_json::from_str(&stats).unwrap();
    assert_eq!(stats["ccr_entry_count"].as_u64(), Some(1));
}

fn run_hr(db_path: &std::path::Path, args: &[&str]) -> String {
    let output = Command::new(env!("CARGO_BIN_EXE_hr"))
        .args(args)
        .env("HR_CCR_DB", db_path)
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    String::from_utf8(output.stdout).unwrap()
}

fn run_hr_with_stdin(db_path: &std::path::Path, args: &[&str], input: &str) -> String {
    let mut child = Command::new(env!("CARGO_BIN_EXE_hr"))
        .args(args)
        .env("HR_CCR_DB", db_path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();

    child
        .stdin
        .as_mut()
        .unwrap()
        .write_all(input.as_bytes())
        .unwrap();

    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    String::from_utf8(output.stdout).unwrap()
}
