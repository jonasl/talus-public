// cfg(test)-only: the run-config surface of the native vision block (see from_json).
use crate::config::ServerConfig;
use crate::json::JsonValue;

#[test]
fn native_vision_configures_the_engine_and_keeps_batch_slots() {
    let cfg = ServerConfig::from_json(
        &JsonValue::parse(
            r#"{"exe":"/x/talus-cli","args":["--batch","4","--slots","2","--max-context","32768"],
                "env":{"STRATA_X":"1"},
                "vision":{"native":true,"mmproj":"/m/mmproj.gguf"}}"#,
        )
        .unwrap(),
    )
    .unwrap();
    assert!(cfg.vision.as_ref().is_some_and(|v| v.native));
    // the tower's weights and the Rust serve loop's handoff go to the engine by environment
    assert!(cfg
        .engine_env
        .iter()
        .any(|(k, v)| k == "TALUS_VISION_MM" && v == "/m/mmproj.gguf"));
    assert!(cfg
        .engine_env
        .iter()
        .any(|(k, v)| k == "TALUS_FN_RUST_VISION" && v == "1"));
    assert!(cfg.engine_env.iter().any(|(k, _)| k == "STRATA_X"));
    // the batch slots stay: with the tower reachable from the C++ loop too, @raw pictures do
    // not force the Rust loop, so a config that asked for slots gets them
    assert!(cfg
        .engine_args
        .iter()
        .any(|a| a == "--batch" || a == "--slots"));
    assert!(cfg.engine_args.iter().any(|a| a == "--max-context"));
    assert!(cfg.engine_args.iter().any(|a| a == "--vision"));
}

#[test]
fn child_vision_config_is_left_alone() {
    let cfg = ServerConfig::from_json(
        &JsonValue::parse(
            r#"{"exe":"/x/talus-cli","args":["--batch","4"],
                "vision":{"exe":"/x/strata-vision-hip","mmproj":"/m/mmproj.gguf"}}"#,
        )
        .unwrap(),
    )
    .unwrap();
    assert!(cfg.vision.as_ref().is_some_and(|v| !v.native));
    assert!(cfg.engine_args.iter().any(|a| a == "--batch"));
    assert!(cfg.engine_args.iter().any(|a| a == "--vision"));
    assert!(!cfg
        .engine_env
        .iter()
        .any(|(k, _)| k == "TALUS_VISION_MM" || k == "TALUS_FN_RUST_VISION"));
}

#[test]
fn an_explicit_env_entry_still_wins_for_the_native_tower() {
    let cfg = ServerConfig::from_json(
        &JsonValue::parse(
            r#"{"exe":"/x/talus-cli","args":[],"env":{"TALUS_VISION_MM":"/m/other.gguf"},
                "vision":{"native":true,"mmproj":"/m/mmproj.gguf"}}"#,
        )
        .unwrap(),
    )
    .unwrap();
    let seen: Vec<&str> = cfg
        .engine_env
        .iter()
        .filter(|(k, _)| k == "TALUS_VISION_MM")
        .map(|(_, v)| v.as_str())
        .collect();
    assert_eq!(seen, vec!["/m/other.gguf"]); // the config's own value; no duplicate injected
}
