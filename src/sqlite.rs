// src/sqlite.rs - SQLite-based IndexedDB shim backend

use deno_core::op2;
use deno_error::JsErrorBox;
use rusqlite::{Connection, params, OptionalExtension};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};

// Global database connections (one per database name)
static DBS: OnceLock<Mutex<HashMap<String, Arc<Mutex<Connection>>>>> = OnceLock::new();

fn get_dbs() -> &'static Mutex<HashMap<String, Arc<Mutex<Connection>>>> {
    DBS.get_or_init(|| Mutex::new(HashMap::new()))
}

fn get_db_path(db_name: &str) -> PathBuf {
    #[cfg(target_os = "android")]
    let data_dir = {
        // Use /data/data/package_name which the app can write to
        let pkg_dir = std::path::PathBuf::from("/data/data/com.yourcompany.combinedapp/files/indexeddb");
        std::fs::create_dir_all(&pkg_dir).ok();
        pkg_dir
    };
    #[cfg(not(target_os = "android"))]
    let data_dir = std::env::current_dir()
        .unwrap_or_default()
        .join("data")
        .join("indexeddb");
    
    std::fs::create_dir_all(&data_dir).ok();
    data_dir.join(format!("{}.sqlite", db_name))
}

fn get_or_create_db(db_name: &str) -> Result<Arc<Mutex<Connection>>, JsErrorBox> {
    let dbs = get_dbs();
    let mut map = dbs.lock().map_err(|e| JsErrorBox::generic(e.to_string()))?;
    
    if let Some(conn) = map.get(db_name) {
        return Ok(conn.clone());
    }
    
    let path = get_db_path(db_name);
    let conn = Connection::open(&path).map_err(|e| JsErrorBox::generic(e.to_string()))?;
    
    // Enable WAL mode for better concurrency
    conn.execute_batch("PRAGMA journal_mode=WAL;").ok();
    
    let arc = Arc::new(Mutex::new(conn));
    map.insert(db_name.to_string(), arc.clone());
    Ok(arc)
}

// ======================= Ops =======================

/// Open/create a database and ensure object stores exist
#[op2]
#[string]
pub fn op_indexeddb_open(
    #[string] db_name: String,
    #[serde] store_names: Vec<String>,
) -> Result<String, JsErrorBox> {
    let conn_arc = get_or_create_db(&db_name)?;
    
    let conn = conn_arc.lock()
        .map_err(|e| JsErrorBox::generic(e.to_string()))?;
    
    // Create a table for each object store
    // Using BLOB for value to store arbitrary binary data (serialized JS values)
    for store_name in &store_names {
        let table_name = sanitize_table_name(store_name);
        conn.execute(
            &format!(
                "CREATE TABLE IF NOT EXISTS {} (
                    key TEXT PRIMARY KEY,
                    value BLOB NOT NULL
                )",
                table_name
            ),
            [],
        ).map_err(|e| JsErrorBox::generic(e.to_string()))?;
    }
    
    Ok(db_name)
}

/// Get a value from an object store
#[op2]
#[buffer]
pub fn op_indexeddb_get(
    #[string] db_name: String,
    #[string] store_name: String,
    #[string] key: String,
) -> Result<Option<Vec<u8>>, JsErrorBox> {
    let conn_arc = get_or_create_db(&db_name)?;
    
    let conn = conn_arc.lock()
        .map_err(|e| JsErrorBox::generic(e.to_string()))?;
    
    let table_name = sanitize_table_name(&store_name);
    
    let result: Option<Vec<u8>> = conn.query_row(
        &format!("SELECT value FROM {} WHERE key = ?1", table_name),
        params![key],
        |row| row.get(0),
    ).optional().map_err(|e| JsErrorBox::generic(e.to_string()))?;
    
    Ok(result)
}

/// Put a value into an object store (insert or replace)
#[op2(fast)]
pub fn op_indexeddb_put(
    #[string] db_name: String,
    #[string] store_name: String,
    #[string] key: String,
    #[buffer] value: &[u8],
) -> Result<(), JsErrorBox> {
    let conn_arc = get_or_create_db(&db_name)?;
    
    let conn = conn_arc.lock()
        .map_err(|e| JsErrorBox::generic(e.to_string()))?;
    
    let table_name = sanitize_table_name(&store_name);
    
    conn.execute(
        &format!(
            "INSERT OR REPLACE INTO {} (key, value) VALUES (?1, ?2)",
            table_name
        ),
        params![key, value],
    ).map_err(|e| JsErrorBox::generic(e.to_string()))?;
    
    Ok(())
}

/// Delete a value from an object store
#[op2(fast)]
pub fn op_indexeddb_delete(
    #[string] db_name: String,
    #[string] store_name: String,
    #[string] key: String,
) -> Result<(), JsErrorBox> {
    let conn_arc = get_or_create_db(&db_name)?;
    
    let conn = conn_arc.lock()
        .map_err(|e| JsErrorBox::generic(e.to_string()))?;
    
    let table_name = sanitize_table_name(&store_name);
    
    conn.execute(
        &format!("DELETE FROM {} WHERE key = ?1", table_name),
        params![key],
    ).map_err(|e| JsErrorBox::generic(e.to_string()))?;
    
    Ok(())
}

/// Get all keys from an object store
#[op2]
#[serde]
pub fn op_indexeddb_get_all_keys(
    #[string] db_name: String,
    #[string] store_name: String,
) -> Result<Vec<String>, JsErrorBox> {
    let conn_arc = get_or_create_db(&db_name)?;
    
    let conn = conn_arc.lock()
        .map_err(|e| JsErrorBox::generic(e.to_string()))?;
    
    let table_name = sanitize_table_name(&store_name);
    
    let mut stmt = conn.prepare(
        &format!("SELECT key FROM {}", table_name)
    ).map_err(|e| JsErrorBox::generic(e.to_string()))?;
    
    let keys: Result<Vec<String>, _> = stmt
        .query_map([], |row| row.get(0))
        .map_err(|e| JsErrorBox::generic(e.to_string()))?
        .collect();
    
    keys.map_err(|e| JsErrorBox::generic(e.to_string()))
}

/// Clear all data from an object store
#[op2(fast)]
pub fn op_indexeddb_clear(
    #[string] db_name: String,
    #[string] store_name: String,
) -> Result<(), JsErrorBox> {
    let conn_arc = get_or_create_db(&db_name)?;
    
    let conn = conn_arc.lock()
        .map_err(|e| JsErrorBox::generic(e.to_string()))?;
    
    let table_name = sanitize_table_name(&store_name);
    
    conn.execute(
        &format!("DELETE FROM {}", table_name),
        [],
    ).map_err(|e| JsErrorBox::generic(e.to_string()))?;
    
    Ok(())
}

/// Check if object store exists
#[op2(fast)]
pub fn op_indexeddb_store_exists(
    #[string] db_name: String,
    #[string] store_name: String,
) -> Result<bool, JsErrorBox> {
    let conn_arc = get_or_create_db(&db_name)?;
    
    let conn = conn_arc.lock()
        .map_err(|e| JsErrorBox::generic(e.to_string()))?;
    
    let table_name = sanitize_table_name(&store_name);
    
    let exists: i32 = conn.query_row(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?1",
        params![table_name],
        |row| row.get(0),
    ).unwrap_or(0);
    
    Ok(exists > 0)
}

// ======================= localStorage Ops =======================

static LS_CONN: OnceLock<Mutex<Connection>> = OnceLock::new();

fn get_ls_conn() -> &'static Mutex<Connection> {
    LS_CONN.get_or_init(|| {
        #[cfg(target_os = "android")]
        let data_dir = {
            let d = std::path::PathBuf::from("/data/data/com.yourcompany.combinedapp/files/localstorage");
            std::fs::create_dir_all(&d).ok();
            d
        };
        #[cfg(not(target_os = "android"))]
        let data_dir = {
            let d = std::env::current_dir()
                .unwrap_or_default()
                .join("data")
                .join("localstorage");
            std::fs::create_dir_all(&d).ok();
            d
        };

        let path = data_dir.join("storage.sqlite");
        let conn = Connection::open(&path).expect("Failed to open localStorage database");
        conn.execute_batch(
            "PRAGMA journal_mode=WAL;
             PRAGMA synchronous=NORMAL;
             CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT NOT NULL);"
        ).expect("Failed to initialize localStorage table");
        Mutex::new(conn)
    })
}

#[op2]
#[string]
pub fn op_localstorage_get(#[string] key: String) -> Result<Option<String>, JsErrorBox> {
    let conn = get_ls_conn().lock().map_err(|e| JsErrorBox::generic(e.to_string()))?;
    let result: Option<String> = conn.query_row(
        "SELECT value FROM kv WHERE key = ?1",
        params![key],
        |row| row.get(0),
    ).optional().map_err(|e| JsErrorBox::generic(e.to_string()))?;
    Ok(result)
}

#[op2(fast)]
pub fn op_localstorage_set(
    #[string] key: String,
    #[string] value: String,
) -> Result<(), JsErrorBox> {
    let conn = get_ls_conn().lock().map_err(|e| JsErrorBox::generic(e.to_string()))?;
    conn.execute(
        "INSERT OR REPLACE INTO kv (key, value) VALUES (?1, ?2)",
        params![key, value],
    ).map_err(|e| JsErrorBox::generic(e.to_string()))?;
    Ok(())
}

#[op2(fast)]
pub fn op_localstorage_remove(#[string] key: String) -> Result<(), JsErrorBox> {
    let conn = get_ls_conn().lock().map_err(|e| JsErrorBox::generic(e.to_string()))?;
    conn.execute("DELETE FROM kv WHERE key = ?1", params![key])
        .map_err(|e| JsErrorBox::generic(e.to_string()))?;
    Ok(())
}

#[op2(fast)]
pub fn op_localstorage_clear() -> Result<(), JsErrorBox> {
    let conn = get_ls_conn().lock().map_err(|e| JsErrorBox::generic(e.to_string()))?;
    conn.execute("DELETE FROM kv", [])
        .map_err(|e| JsErrorBox::generic(e.to_string()))?;
    Ok(())
}

#[op2]
#[serde]
pub fn op_localstorage_keys() -> Result<Vec<String>, JsErrorBox> {
    let conn = get_ls_conn().lock().map_err(|e| JsErrorBox::generic(e.to_string()))?;
    let mut stmt = conn.prepare("SELECT key FROM kv ORDER BY rowid")
        .map_err(|e| JsErrorBox::generic(e.to_string()))?;
    let keys: Result<Vec<String>, _> = stmt
        .query_map([], |row| row.get(0))
        .map_err(|e| JsErrorBox::generic(e.to_string()))?
        .collect();
    keys.map_err(|e| JsErrorBox::generic(e.to_string()))
}

#[op2(fast)]
pub fn op_localstorage_length() -> Result<u32, JsErrorBox> {
    let conn = get_ls_conn().lock().map_err(|e| JsErrorBox::generic(e.to_string()))?;
    let count: u32 = conn.query_row("SELECT COUNT(*) FROM kv", [], |row| row.get(0))
        .map_err(|e| JsErrorBox::generic(e.to_string()))?;
    Ok(count)
}

// ======================= Helpers =======================

// Helper to sanitize table names (prevent SQL injection)
fn sanitize_table_name(name: &str) -> String {
    // Only allow alphanumeric and underscore
    let sanitized: String = name
        .chars()
        .map(|c| if c.is_alphanumeric() || c == '_' { c } else { '_' })
        .collect();
    
    // Prefix with 'store_' to avoid reserved words
    format!("store_{}", sanitized)
}