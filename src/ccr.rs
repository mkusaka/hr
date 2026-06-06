use crate::{error, stats, HrResult};
use regex::Regex;
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::Path;
use std::sync::{Arc, LazyLock, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

pub const HASH_HEX_LEN: usize = 24;

static MARKER_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"<<ccr:([A-Fa-f0-9]{24,64})>>").expect("valid marker regex"));

pub trait CcrStore: Send + Sync {
    fn put(&self, hash: &str, original: &str) -> HrResult<bool>;
    fn get(&self, hash: &str) -> HrResult<Option<String>>;
    fn count(&self) -> HrResult<u64>;
}

#[derive(Debug, Clone)]
pub struct SqliteStore {
    conn: Arc<Mutex<Connection>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DecompressResult {
    pub output: String,
    pub hits: usize,
    pub misses: usize,
    pub missing_hashes: Vec<String>,
}

pub fn content_hash(input: &str) -> String {
    let digest = Sha256::digest(input.as_bytes());
    digest[..HASH_HEX_LEN / 2]
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

pub fn marker_for_hash(hash: &str) -> String {
    format!("<<ccr:{hash}>>")
}

pub fn decompress_hash(hash: &str, store: &dyn CcrStore) -> Option<String> {
    match store.get(hash) {
        Ok(Some(value)) => {
            stats::record_decompress_hit();
            Some(value)
        }
        Ok(None) | Err(_) => {
            stats::record_decompress_miss();
            None
        }
    }
}

pub fn decompress_text(text: &str, store: &dyn CcrStore) -> DecompressResult {
    let mut output = String::with_capacity(text.len());
    let mut last = 0;
    let mut hits = 0;
    let mut misses = 0;
    let mut missing_hashes = Vec::new();

    for captures in MARKER_RE.captures_iter(text) {
        let Some(marker) = captures.get(0) else {
            continue;
        };
        let Some(hash) = captures.get(1).map(|m| m.as_str()) else {
            continue;
        };

        output.push_str(&text[last..marker.start()]);
        if let Some(original) = decompress_hash(hash, store) {
            output.push_str(&original);
            hits += 1;
        } else {
            output.push_str(marker.as_str());
            misses += 1;
            missing_hashes.push(hash.to_string());
        }
        last = marker.end();
    }

    output.push_str(&text[last..]);

    DecompressResult {
        output,
        hits,
        misses,
        missing_hashes,
    }
}

impl SqliteStore {
    pub fn open(path: impl AsRef<Path>) -> HrResult<Self> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }

        let conn = Connection::open(path)?;
        let store = Self {
            conn: Arc::new(Mutex::new(conn)),
        };
        store.init()?;
        Ok(store)
    }

    pub fn in_memory() -> HrResult<Self> {
        let store = Self {
            conn: Arc::new(Mutex::new(Connection::open_in_memory()?)),
        };
        store.init()?;
        Ok(store)
    }

    fn init(&self) -> HrResult<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|_| error("sqlite connection lock poisoned"))?;
        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS ccr_entries (
                hash TEXT PRIMARY KEY,
                original TEXT NOT NULL,
                bytes INTEGER NOT NULL,
                tokens INTEGER NOT NULL,
                created_at INTEGER NOT NULL,
                last_seen_at INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_ccr_entries_created_at
                ON ccr_entries(created_at);
            "#,
        )?;
        Ok(())
    }
}

impl CcrStore for SqliteStore {
    fn put(&self, hash: &str, original: &str) -> HrResult<bool> {
        let now = unix_timestamp();
        let conn = self
            .conn
            .lock()
            .map_err(|_| error("sqlite connection lock poisoned"))?;
        let changed = conn.execute(
            r#"
            INSERT OR IGNORE INTO ccr_entries
                (hash, original, bytes, tokens, created_at, last_seen_at)
            VALUES (?1, ?2, ?3, ?4, ?5, ?5)
            "#,
            params![
                hash,
                original,
                original.len() as i64,
                crate::compression::estimate_tokens(original) as i64,
                now
            ],
        )?;

        if changed == 0 {
            conn.execute(
                "UPDATE ccr_entries SET last_seen_at = ?1 WHERE hash = ?2",
                params![now, hash],
            )?;
        }

        Ok(changed > 0)
    }

    fn get(&self, hash: &str) -> HrResult<Option<String>> {
        let conn = self
            .conn
            .lock()
            .map_err(|_| error("sqlite connection lock poisoned"))?;
        let value = conn
            .query_row(
                "SELECT original FROM ccr_entries WHERE hash = ?1",
                params![hash],
                |row| row.get(0),
            )
            .optional()?;
        Ok(value)
    }

    fn count(&self) -> HrResult<u64> {
        let conn = self
            .conn
            .lock()
            .map_err(|_| error("sqlite connection lock poisoned"))?;
        let count: i64 =
            conn.query_row("SELECT COUNT(*) FROM ccr_entries", [], |row| row.get(0))?;
        Ok(count.max(0) as u64)
    }
}

fn unix_timestamp() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs() as i64)
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ccr_hash_and_marker_are_deterministic() {
        let hash = content_hash("hello");

        assert_eq!(hash, content_hash("hello"));
        assert_eq!(hash.len(), HASH_HEX_LEN);
        assert_eq!(marker_for_hash(&hash), format!("<<ccr:{hash}>>"));
    }

    #[test]
    fn sqlite_put_get_and_count() {
        let store = SqliteStore::in_memory().unwrap();
        let hash = content_hash("payload");

        assert!(store.put(&hash, "payload").unwrap());
        assert!(!store.put(&hash, "payload").unwrap());

        assert_eq!(store.get(&hash).unwrap(), Some("payload".to_string()));
        assert_eq!(store.count().unwrap(), 1);
    }

    #[test]
    fn decompress_hash_and_text_expand_markers() {
        crate::stats::reset_for_tests();
        let store = SqliteStore::in_memory().unwrap();
        let hash = content_hash("original");
        store.put(&hash, "original").unwrap();

        assert_eq!(decompress_hash(&hash, &store), Some("original".to_string()));

        let result = decompress_text(
            &format!(
                "before {} after <<ccr:000000000000000000000000>>",
                marker_for_hash(&hash)
            ),
            &store,
        );

        assert_eq!(
            result.output,
            "before original after <<ccr:000000000000000000000000>>"
        );
        assert_eq!(result.hits, 1);
        assert_eq!(result.misses, 1);
        assert_eq!(result.missing_hashes, vec!["000000000000000000000000"]);
    }
}
