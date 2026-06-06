use clap::{ArgGroup, Parser, Subcommand};
use hr::ccr::CcrStore;
use hr::{
    compress, decompress_hash, decompress_text, serve_proxy, CompressOptions, CompressionMode,
    HrResult, ProxyConfig, SqliteStore, DEFAULT_MAX_BODY_BYTES,
};
use std::io::Read;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use tracing_subscriber::EnvFilter;
use url::Url;

#[derive(Debug, Parser)]
#[command(
    name = "hr",
    version,
    about = "Small Rust core for CCR compression and proxying"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Compress {
        #[arg(long)]
        input: PathBuf,
    },
    #[command(group(
        ArgGroup::new("source")
            .required(true)
            .args(["hash", "input"])
    ))]
    Decompress {
        #[arg(long)]
        hash: Option<String>,
        #[arg(long)]
        input: Option<PathBuf>,
    },
    Proxy {
        #[arg(long)]
        listen: SocketAddr,
        #[arg(long)]
        openai_upstream: String,
        #[arg(long)]
        anthropic_upstream: String,
        #[arg(long)]
        ccr_db: PathBuf,
        #[arg(long, default_value = "info")]
        log_level: String,
        #[arg(long, alias = "compression-max-body-bytes", default_value_t = DEFAULT_MAX_BODY_BYTES)]
        max_body_bytes: usize,
        #[arg(long, default_value_t = true)]
        compression: bool,
        #[arg(long, default_value = "ccr")]
        compression_mode: String,
    },
    Stats,
}

#[tokio::main]
async fn main() -> HrResult<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::Compress { input } => {
            let input = read_input(&input)?;
            let store = SqliteStore::open(default_ccr_db())?;
            let result = compress(
                &input,
                CompressOptions {
                    store: &store,
                    min_bytes: 1,
                },
            );

            if let Some(error) = result.error {
                return Err(hr::error(format!("compression failed: {error}")));
            }

            println!("{}", result.output);
        }
        Command::Decompress { hash, input } => {
            let store = SqliteStore::open(default_ccr_db())?;
            if let Some(hash) = hash {
                let Some(original) = decompress_hash(&hash, &store) else {
                    return Err(hr::error(format!("hash not found: {hash}")));
                };
                println!("{original}");
            } else if let Some(input) = input {
                let input = read_input(&input)?;
                let result = decompress_text(&input, &store);
                println!("{}", result.output);
            }
        }
        Command::Proxy {
            listen,
            openai_upstream,
            anthropic_upstream,
            ccr_db,
            log_level,
            max_body_bytes,
            compression,
            compression_mode,
        } => {
            init_tracing(&log_level)?;
            let openai_upstream = Url::parse(&openai_upstream)?;
            let anthropic_upstream = Url::parse(&anthropic_upstream)?;
            let compression_mode = CompressionMode::parse(&compression_mode)?;
            serve_proxy(ProxyConfig {
                listen,
                openai_upstream,
                anthropic_upstream,
                ccr_db,
                log_level,
                max_body_bytes,
                compression_enabled: compression,
                compression_mode,
            })
            .await?;
        }
        Command::Stats => {
            let store = SqliteStore::open(default_ccr_db())?;
            let snapshot = hr::stats::stats_with_ccr_entry_count(store.count()?);
            println!("{}", serde_json::to_string_pretty(&snapshot)?);
        }
    }

    Ok(())
}

fn read_input(path: &Path) -> HrResult<String> {
    if path == Path::new("-") {
        let mut input = String::new();
        std::io::stdin().read_to_string(&mut input)?;
        Ok(input)
    } else {
        Ok(std::fs::read_to_string(path)?)
    }
}

fn default_ccr_db() -> PathBuf {
    std::env::var_os("HR_CCR_DB")
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".hr").join("ccr.sqlite"))
        })
        .unwrap_or_else(|| PathBuf::from(".hr").join("ccr.sqlite"))
}

fn init_tracing(log_level: &str) -> HrResult<()> {
    let filter = EnvFilter::try_new(log_level)?;
    tracing_subscriber::fmt()
        .json()
        .with_env_filter(filter)
        .try_init()
        .map_err(|err| hr::error(format!("failed to initialize tracing: {err}")))?;
    Ok(())
}
