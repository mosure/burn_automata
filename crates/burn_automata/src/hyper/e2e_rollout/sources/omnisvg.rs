use std::{collections::BTreeMap, io::Read, time::Duration};

use base64::{Engine as _, engine::general_purpose};

use super::{Hyper2dScratchSource, OmniSvgDataset, sanitize_slug};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

const HF_DATASET_ROWS_URL: &str = "https://datasets-server.huggingface.co/rows";
const HF_DATASET_CONFIG: &str = "default";
const MANIFEST_VERSION: u32 = 1;
const HTTP_USER_AGENT: &str = "burn_automata/0.1 omnisvg-thumbnail-loader";
const HTTP_FETCH_ATTEMPTS: usize = 12;
const HF_ROWS_FETCH_ATTEMPTS: usize = 24;
const HTTP_FETCH_MAX_BACKOFF: Duration = Duration::from_secs(30);
const HTTP_REQUEST_TIMEOUT: Duration = Duration::from_secs(120);
const THUMBNAIL_FETCH_PARALLELISM: usize = 8;

#[derive(Clone, Copy, Debug)]
pub(crate) struct OmniSvgSourceConfig<'a> {
    pub(crate) dataset: OmniSvgDataset,
    pub(crate) split: &'a str,
    pub(crate) cache_dir: &'a Path,
    pub(crate) offset: usize,
    pub(crate) limit: usize,
    pub(crate) page_size: usize,
    pub(crate) download: bool,
    pub(crate) refresh: bool,
    pub(crate) token_env: &'a str,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct OmniSvgCacheManifest {
    version: u32,
    dataset: String,
    split: String,
    entries: Vec<OmniSvgCacheEntry>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct OmniSvgCacheEntry {
    id: String,
    slug: String,
    title: Option<String>,
    description: Option<String>,
    keywords: Option<String>,
    detail: Option<String>,
    thumbnail_file: String,
    source_offset: usize,
    source_url: Option<String>,
    width: Option<u32>,
    height: Option<u32>,
}

#[derive(Clone, Debug)]
struct OmniSvgCachePaths {
    root: PathBuf,
    manifest: PathBuf,
}

#[derive(Clone, Debug)]
struct ParsedOmniSvgRow {
    id: String,
    slug: String,
    title: Option<String>,
    description: Option<String>,
    keywords: Option<String>,
    detail: Option<String>,
    image: OmniSvgImagePayload,
    source_offset: usize,
    width: Option<u32>,
    height: Option<u32>,
}

#[derive(Clone, Debug)]
struct ParsedOmniSvgImage {
    payload: OmniSvgImagePayload,
    width: Option<u32>,
    height: Option<u32>,
}

#[derive(Clone, Debug)]
enum OmniSvgImagePayload {
    Url(String),
    Bytes(Vec<u8>),
}

#[derive(Debug, Deserialize)]
struct HfRowsResponse {
    rows: Vec<HfRowsEntry>,
}

#[derive(Debug, Deserialize)]
struct HfRowsEntry {
    row: serde_json::Value,
}

pub(super) fn resolve_omnisvg_scratch_sources(
    config: OmniSvgSourceConfig<'_>,
) -> Result<Vec<Hyper2dScratchSource>, Box<dyn std::error::Error>> {
    validate_omnisvg_config(config)?;
    let paths = OmniSvgCachePaths::new(config.cache_dir, config.dataset, config.split);
    let mut manifest = load_or_create_manifest(&paths, config.dataset, config.split)?;
    if config.download {
        manifest = ensure_omnisvg_cache(config, &paths, manifest)?;
    }

    let requested_end = config
        .offset
        .checked_add(config.limit)
        .ok_or_else(|| std::io::Error::other("OmniSVG offset + limit overflowed"))?;
    let mut entries = manifest
        .entries
        .iter()
        .filter(|entry| entry.source_offset >= config.offset && entry.source_offset < requested_end)
        .filter(|entry| cached_thumbnail_exists(&paths, entry))
        .cloned()
        .collect::<Vec<_>>();
    entries.sort_by_key(|entry| entry.source_offset);
    entries.dedup_by_key(|entry| entry.source_offset);

    if entries.len() < config.limit {
        return Err(std::io::Error::other(format!(
            "OmniSVG cache has {} usable thumbnails for requested range [{}..{}); rerun with --omnisvg-download=true or lower --omnisvg-limit",
            entries.len(),
            config.offset,
            requested_end
        ))
        .into());
    }

    Ok(entries
        .into_iter()
        .take(config.limit)
        .map(|entry| Hyper2dScratchSource {
            slug: entry.slug,
            title: entry.title,
            group: Some(format!("omnisvg:{}", config.dataset.cache_slug())),
            condition_path: paths.root.join(entry.thumbnail_file),
            particles: None,
            seed_scale: None,
            update_prob: None,
        })
        .collect())
}

fn validate_omnisvg_config(config: OmniSvgSourceConfig<'_>) -> Result<(), std::io::Error> {
    if config.limit == 0 {
        return Err(std::io::Error::other(
            "--omnisvg-limit must be greater than zero",
        ));
    }
    if config.page_size == 0 {
        return Err(std::io::Error::other(
            "--omnisvg-page-size must be greater than zero",
        ));
    }
    if config.split.trim().is_empty() {
        return Err(std::io::Error::other("--omnisvg-split cannot be empty"));
    }
    Ok(())
}

fn ensure_omnisvg_cache(
    config: OmniSvgSourceConfig<'_>,
    paths: &OmniSvgCachePaths,
    manifest: OmniSvgCacheManifest,
) -> Result<OmniSvgCacheManifest, Box<dyn std::error::Error>> {
    std::fs::create_dir_all(&paths.root)?;
    let token = read_hf_token(config.token_env);
    let requested_end = config
        .offset
        .checked_add(config.limit)
        .ok_or_else(|| std::io::Error::other("OmniSVG offset + limit overflowed"))?;
    let mut entries_by_offset = manifest
        .entries
        .into_iter()
        .map(|entry| (entry.source_offset, entry))
        .collect::<BTreeMap<_, _>>();

    let mut page_offset = config.offset;
    while page_offset < requested_end {
        let page_len = config.page_size.min(requested_end - page_offset);
        let needs_page = config.refresh
            || (page_offset..page_offset + page_len).any(|offset| {
                entries_by_offset
                    .get(&offset)
                    .is_none_or(|entry| !cached_thumbnail_exists(paths, entry))
            });
        if needs_page {
            maybe_log_omnisvg_cache_progress(
                "fetching",
                page_offset,
                page_len,
                config.offset,
                requested_end,
            );
            let response_text = fetch_hf_rows_page(
                config.dataset.dataset_id(),
                config.split,
                page_offset,
                page_len,
                token.as_deref(),
            )?;
            let rows = parse_omnisvg_rows_response(&response_text, page_offset)?;
            for entry in cache_omnisvg_page_rows(
                paths,
                rows,
                requested_end,
                config.refresh,
                token.as_deref(),
            )? {
                entries_by_offset.insert(entry.source_offset, entry);
            }
            save_entries_manifest(paths, config.dataset, config.split, &entries_by_offset)?;
            maybe_log_omnisvg_cache_progress(
                "cached",
                page_offset,
                page_len,
                config.offset,
                requested_end,
            );
        }
        page_offset += page_len;
    }

    let manifest = entries_manifest(config.dataset, config.split, &entries_by_offset);
    save_manifest(paths, &manifest)?;
    Ok(manifest)
}

fn cache_omnisvg_page_rows(
    paths: &OmniSvgCachePaths,
    rows: Vec<ParsedOmniSvgRow>,
    requested_end: usize,
    refresh: bool,
    token: Option<&str>,
) -> Result<Vec<OmniSvgCacheEntry>, Box<dyn std::error::Error>> {
    let mut entries = Vec::with_capacity(rows.len());
    for chunk in rows.chunks(THUMBNAIL_FETCH_PARALLELISM) {
        let root = &paths.root;
        let chunk_entries =
            std::thread::scope(|scope| -> Result<Vec<OmniSvgCacheEntry>, std::io::Error> {
                let mut handles = Vec::with_capacity(chunk.len());
                for row in chunk.iter().cloned() {
                    handles.push(scope.spawn(move || {
                        cache_omnisvg_row(root, row, requested_end, refresh, token)
                    }));
                }

                let mut chunk_entries = Vec::with_capacity(chunk.len());
                for handle in handles {
                    if let Some(entry) = handle
                        .join()
                        .map_err(|_| std::io::Error::other("OmniSVG thumbnail worker panicked"))??
                    {
                        chunk_entries.push(entry);
                    }
                }
                Ok(chunk_entries)
            })?;
        entries.extend(chunk_entries);
    }
    Ok(entries)
}

fn cache_omnisvg_row(
    root: &Path,
    row: ParsedOmniSvgRow,
    requested_end: usize,
    refresh: bool,
    token: Option<&str>,
) -> Result<Option<OmniSvgCacheEntry>, std::io::Error> {
    if row.source_offset >= requested_end {
        return Ok(None);
    }
    let thumbnail_file = thumbnail_file_name(row.source_offset, &row.id, &row.image);
    let thumbnail_path = root.join(&thumbnail_file);
    if refresh || !thumbnail_path.exists() {
        let bytes = image_bytes(row.image.clone(), token)
            .map_err(|err| std::io::Error::other(format!("{err}")))?;
        if bytes.is_empty() {
            return Err(std::io::Error::other(format!(
                "OmniSVG thumbnail for row {} was empty",
                row.source_offset
            )));
        }
        let tmp_path = root.join(format!("{thumbnail_file}.part"));
        std::fs::write(&tmp_path, bytes)?;
        std::fs::rename(&tmp_path, &thumbnail_path)?;
    }
    Ok(Some(OmniSvgCacheEntry {
        id: row.id,
        slug: row.slug,
        title: row.title,
        description: row.description,
        keywords: row.keywords,
        detail: row.detail,
        thumbnail_file,
        source_offset: row.source_offset,
        source_url: row.image.source_url(),
        width: row.width,
        height: row.height,
    }))
}

fn maybe_log_omnisvg_cache_progress(
    phase: &str,
    page_offset: usize,
    page_len: usize,
    requested_offset: usize,
    requested_end: usize,
) {
    let page_end = page_offset.saturating_add(page_len);
    let page_index = page_offset.saturating_sub(requested_offset) / page_len.max(1);
    let page_count = requested_end
        .saturating_sub(requested_offset)
        .div_ceil(page_len.max(1));
    if page_index == 0 || page_index.is_multiple_of(10) || page_end >= requested_end {
        eprintln!(
            "OmniSVG cache {phase} rows {page_offset}..{page_end} ({}/{page_count} pages)",
            page_index + 1
        );
    }
}

fn load_or_create_manifest(
    paths: &OmniSvgCachePaths,
    dataset: OmniSvgDataset,
    split: &str,
) -> Result<OmniSvgCacheManifest, Box<dyn std::error::Error>> {
    if !paths.manifest.exists() {
        return Ok(OmniSvgCacheManifest {
            version: MANIFEST_VERSION,
            dataset: dataset.dataset_id().to_string(),
            split: split.to_string(),
            entries: Vec::new(),
        });
    }
    let text = std::fs::read_to_string(&paths.manifest)?;
    let manifest: OmniSvgCacheManifest = serde_json::from_str(&text)?;
    if manifest.dataset != dataset.dataset_id() || manifest.split != split {
        return Err(std::io::Error::other(format!(
            "OmniSVG cache manifest {:?} is for dataset={} split={}, not dataset={} split={}",
            paths.manifest,
            manifest.dataset,
            manifest.split,
            dataset.dataset_id(),
            split
        ))
        .into());
    }
    Ok(manifest)
}

fn save_manifest(
    paths: &OmniSvgCachePaths,
    manifest: &OmniSvgCacheManifest,
) -> Result<(), Box<dyn std::error::Error>> {
    std::fs::create_dir_all(&paths.root)?;
    let tmp_manifest = paths.manifest.with_file_name("manifest.json.part");
    std::fs::write(&tmp_manifest, serde_json::to_string_pretty(manifest)?)?;
    std::fs::rename(&tmp_manifest, &paths.manifest)?;
    Ok(())
}

fn save_entries_manifest(
    paths: &OmniSvgCachePaths,
    dataset: OmniSvgDataset,
    split: &str,
    entries_by_offset: &BTreeMap<usize, OmniSvgCacheEntry>,
) -> Result<(), Box<dyn std::error::Error>> {
    let manifest = entries_manifest(dataset, split, entries_by_offset);
    save_manifest(paths, &manifest)
}

fn entries_manifest(
    dataset: OmniSvgDataset,
    split: &str,
    entries_by_offset: &BTreeMap<usize, OmniSvgCacheEntry>,
) -> OmniSvgCacheManifest {
    OmniSvgCacheManifest {
        version: MANIFEST_VERSION,
        dataset: dataset.dataset_id().to_string(),
        split: split.to_string(),
        entries: entries_by_offset.values().cloned().collect(),
    }
}

fn fetch_hf_rows_page(
    dataset_id: &str,
    split: &str,
    offset: usize,
    length: usize,
    token: Option<&str>,
) -> Result<String, Box<dyn std::error::Error>> {
    let url = hf_rows_url(dataset_id, split, offset, length);
    let bytes = http_get_bytes_with_attempts(&url, token, HF_ROWS_FETCH_ATTEMPTS)?;
    Ok(String::from_utf8(bytes)?)
}

fn parse_omnisvg_rows_response(
    text: &str,
    page_offset: usize,
) -> Result<Vec<ParsedOmniSvgRow>, Box<dyn std::error::Error>> {
    let response: HfRowsResponse = serde_json::from_str(text)?;
    response
        .rows
        .into_iter()
        .enumerate()
        .map(|(index, entry)| parse_omnisvg_row(entry.row, page_offset + index))
        .collect()
}

fn parse_omnisvg_row(
    row: serde_json::Value,
    source_offset: usize,
) -> Result<ParsedOmniSvgRow, Box<dyn std::error::Error>> {
    let row = row.as_object().ok_or_else(|| {
        std::io::Error::other(format!("OmniSVG row {source_offset} was not a JSON object"))
    })?;
    let id = row
        .get("id")
        .and_then(json_scalar_to_string)
        .unwrap_or_else(|| format!("row-{source_offset}"));
    let description = row.get("description").and_then(json_scalar_to_string);
    let keywords = row.get("keywords").and_then(json_scalar_to_string);
    let detail = row.get("detail").and_then(json_scalar_to_string);
    let image = row.get("image").ok_or_else(|| {
        std::io::Error::other(format!("OmniSVG row {source_offset} did not contain image"))
    })?;
    let image = parse_image_payload(image, source_offset)?;
    let slug = format!("{source_offset:08}_{}", sanitize_slug(&id));
    let title = description.as_deref().map(compact_title);
    Ok(ParsedOmniSvgRow {
        id,
        slug,
        title,
        description,
        keywords,
        detail,
        image: image.payload,
        source_offset,
        width: image.width,
        height: image.height,
    })
}

fn parse_image_payload(
    value: &serde_json::Value,
    source_offset: usize,
) -> Result<ParsedOmniSvgImage, Box<dyn std::error::Error>> {
    if let Some(url_or_data) = value.as_str() {
        return Ok(ParsedOmniSvgImage {
            payload: image_payload_from_string(url_or_data)?,
            width: None,
            height: None,
        });
    }
    let image = value.as_object().ok_or_else(|| {
        std::io::Error::other(format!(
            "OmniSVG row {source_offset} image was not an object or string"
        ))
    })?;
    let width = image
        .get("width")
        .and_then(serde_json::Value::as_u64)
        .and_then(|value| u32::try_from(value).ok());
    let height = image
        .get("height")
        .and_then(serde_json::Value::as_u64)
        .and_then(|value| u32::try_from(value).ok());
    if let Some(src) = image.get("src").and_then(serde_json::Value::as_str) {
        return Ok(ParsedOmniSvgImage {
            payload: OmniSvgImagePayload::Url(src.to_string()),
            width,
            height,
        });
    }
    if let Some(bytes) = image.get("bytes").and_then(serde_json::Value::as_str) {
        return Ok(ParsedOmniSvgImage {
            payload: OmniSvgImagePayload::Bytes(decode_base64_image(bytes)?),
            width,
            height,
        });
    }
    Err(std::io::Error::other(format!(
        "OmniSVG row {source_offset} image did not contain src or bytes"
    ))
    .into())
}

fn image_payload_from_string(
    value: &str,
) -> Result<OmniSvgImagePayload, Box<dyn std::error::Error>> {
    if value.starts_with("data:image/") {
        return Ok(OmniSvgImagePayload::Bytes(decode_base64_image(value)?));
    }
    Ok(OmniSvgImagePayload::Url(value.to_string()))
}

fn image_bytes(
    payload: OmniSvgImagePayload,
    token: Option<&str>,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    match payload {
        OmniSvgImagePayload::Url(url) => http_get_bytes(&url, token),
        OmniSvgImagePayload::Bytes(bytes) => Ok(bytes),
    }
}

fn http_get_bytes(url: &str, token: Option<&str>) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    http_get_bytes_with_attempts(url, token, HTTP_FETCH_ATTEMPTS)
}

fn http_get_bytes_with_attempts(
    url: &str,
    token: Option<&str>,
    max_attempts: usize,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let mut delay = Duration::from_millis(500);
    for attempt in 1..=max_attempts {
        match http_get_bytes_once(url, token) {
            Ok(bytes) => return Ok(bytes),
            Err(err) if attempt == max_attempts => return Err(err),
            Err(err) => {
                let log_url = redacted_http_url(url);
                eprintln!(
                    "warning: HTTP fetch attempt {attempt}/{max_attempts} failed for {log_url}: {err}; retrying in {} ms",
                    delay.as_millis()
                );
                std::thread::sleep(delay);
                delay = delay.saturating_mul(2).min(HTTP_FETCH_MAX_BACKOFF);
            }
        }
    }
    unreachable!("HTTP retry loop always returns")
}

fn http_get_bytes_once(
    url: &str,
    token: Option<&str>,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let mut request = ureq::get(url)
        .timeout(HTTP_REQUEST_TIMEOUT)
        .set("User-Agent", HTTP_USER_AGENT)
        .set("Accept", "*/*");
    let bearer;
    if let Some(token) = token {
        bearer = format!("Bearer {token}");
        request = request.set("Authorization", &bearer);
    }
    let log_url = redacted_http_url(url);
    let response = request.call().map_err(|err| {
        std::io::Error::other(format!(
            "HTTP fetch failed for {log_url}: {}",
            redacted_http_error(err, url)
        ))
    })?;
    let mut bytes = Vec::new();
    response
        .into_reader()
        .read_to_end(&mut bytes)
        .map_err(|err| {
            std::io::Error::other(format!("HTTP body read failed for {log_url}: {err}"))
        })?;
    Ok(bytes)
}

fn redacted_http_error(err: ureq::Error, url: &str) -> String {
    format!("{err}").replace(url, &redacted_http_url(url))
}

fn redacted_http_url(url: &str) -> String {
    let mut redacted = url
        .split_once('?')
        .map_or(url, |(base, _)| base)
        .to_string();
    const MAX_LOG_URL_LEN: usize = 160;
    if redacted.len() > MAX_LOG_URL_LEN {
        redacted.truncate(MAX_LOG_URL_LEN);
        redacted.push_str("...");
    }
    if url.contains('?') {
        redacted.push_str("?...");
    }
    redacted
}

fn decode_base64_image(value: &str) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let encoded = value
        .split_once(',')
        .map_or(value, |(_, encoded)| encoded)
        .trim();
    general_purpose::STANDARD
        .decode(encoded)
        .or_else(|_| general_purpose::URL_SAFE.decode(encoded))
        .map_err(|err| std::io::Error::other(format!("invalid base64 image payload: {err}")).into())
}

fn read_hf_token(token_env: &str) -> Option<String> {
    if token_env.trim().is_empty() {
        return None;
    }
    std::env::var(token_env)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn hf_rows_url(dataset_id: &str, split: &str, offset: usize, length: usize) -> String {
    format!(
        "{HF_DATASET_ROWS_URL}?dataset={}&config={HF_DATASET_CONFIG}&split={}&offset={offset}&length={length}",
        query_escape(dataset_id),
        query_escape(split)
    )
}

fn query_escape(value: &str) -> String {
    value
        .bytes()
        .map(|byte| match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' => {
                (byte as char).to_string()
            }
            _ => format!("%{byte:02X}"),
        })
        .collect()
}

fn json_scalar_to_string(value: &serde_json::Value) -> Option<String> {
    match value {
        serde_json::Value::String(value) => Some(value.clone()),
        serde_json::Value::Number(value) => Some(value.to_string()),
        serde_json::Value::Bool(value) => Some(value.to_string()),
        _ => None,
    }
}

fn compact_title(description: &str) -> String {
    let value = description.split_whitespace().collect::<Vec<_>>().join(" ");
    const MAX_TITLE_LEN: usize = 96;
    if value.chars().count() <= MAX_TITLE_LEN {
        value
    } else {
        format!(
            "{}...",
            value.chars().take(MAX_TITLE_LEN).collect::<String>()
        )
    }
}

fn thumbnail_file_name(offset: usize, id: &str, payload: &OmniSvgImagePayload) -> String {
    let lower_url = match payload {
        OmniSvgImagePayload::Url(url) => Some(url.to_ascii_lowercase()),
        OmniSvgImagePayload::Bytes(_) => None,
    };
    let extension = if lower_url
        .as_deref()
        .is_some_and(|url| url.contains(".jpg") || url.contains(".jpeg"))
    {
        "jpg"
    } else if lower_url
        .as_deref()
        .is_some_and(|url| url.contains(".webp"))
    {
        "webp"
    } else {
        "png"
    };
    format!("{offset:08}_{}.{}", sanitize_slug(id), extension)
}

fn cached_thumbnail_exists(paths: &OmniSvgCachePaths, entry: &OmniSvgCacheEntry) -> bool {
    paths
        .root
        .join(&entry.thumbnail_file)
        .metadata()
        .is_ok_and(|metadata| metadata.is_file() && metadata.len() > 0)
}

impl OmniSvgCachePaths {
    fn new(cache_dir: &Path, dataset: OmniSvgDataset, split: &str) -> Self {
        let root = cache_dir
            .join(dataset.cache_slug())
            .join(sanitize_slug(split));
        let manifest = root.join("manifest.json");
        Self { root, manifest }
    }
}

impl OmniSvgImagePayload {
    fn source_url(&self) -> Option<String> {
        match self {
            Self::Url(url) => Some(url.clone()),
            Self::Bytes(_) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_parser_extracts_thumbnail_src_and_metadata() {
        let json = r#"{
            "rows": [{
                "row": {
                    "id": "abc-123",
                    "description": "A compact vector lizard thumbnail.",
                    "keywords": "lizard, vector",
                    "detail": "green shape",
                    "image": {
                        "src": "https://datasets-server.huggingface.co/cached-assets/example/image.png",
                        "width": 448,
                        "height": 448
                    }
                }
            }]
        }"#;

        let rows = parse_omnisvg_rows_response(json, 42).unwrap();

        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].id, "abc-123");
        assert_eq!(rows[0].slug, "00000042_abc-123");
        assert_eq!(
            rows[0].title.as_deref(),
            Some("A compact vector lizard thumbnail.")
        );
        assert_eq!(rows[0].keywords.as_deref(), Some("lizard, vector"));
        assert_eq!(rows[0].width, Some(448));
        assert_eq!(rows[0].height, Some(448));
        assert!(matches!(rows[0].image, OmniSvgImagePayload::Url(_)));
    }

    #[test]
    fn rows_parser_accepts_base64_image_payloads() {
        let json = r#"{
            "rows": [{
                "row": {
                    "id": "inline",
                    "image": {
                        "bytes": "iVBORw0KGgo=",
                        "width": 1,
                        "height": 1
                    }
                }
            }]
        }"#;

        let rows = parse_omnisvg_rows_response(json, 0).unwrap();

        let OmniSvgImagePayload::Bytes(bytes) = &rows[0].image else {
            panic!("expected inline bytes");
        };
        assert_eq!(bytes, b"\x89PNG\r\n\x1a\n");
    }

    #[test]
    fn cache_only_resolver_reads_manifest_range() {
        let cache_dir = unique_temp_dir("burn_automata_omnisvg_cache_only");
        let paths = OmniSvgCachePaths::new(&cache_dir, OmniSvgDataset::MmsvgIllustration, "train");
        std::fs::create_dir_all(&paths.root).unwrap();
        std::fs::write(paths.root.join("00000005_abcd.png"), b"png").unwrap();
        let manifest = OmniSvgCacheManifest {
            version: MANIFEST_VERSION,
            dataset: OmniSvgDataset::MmsvgIllustration.dataset_id().to_string(),
            split: "train".to_string(),
            entries: vec![OmniSvgCacheEntry {
                id: "abcd".to_string(),
                slug: "00000005_abcd".to_string(),
                title: Some("cached preview".to_string()),
                description: None,
                keywords: None,
                detail: None,
                thumbnail_file: "00000005_abcd.png".to_string(),
                source_offset: 5,
                source_url: Some("https://example.invalid/thumb.png".to_string()),
                width: Some(448),
                height: Some(448),
            }],
        };
        save_manifest(&paths, &manifest).unwrap();

        let sources = resolve_omnisvg_scratch_sources(OmniSvgSourceConfig {
            dataset: OmniSvgDataset::MmsvgIllustration,
            split: "train",
            cache_dir: &cache_dir,
            offset: 5,
            limit: 1,
            page_size: 100,
            download: false,
            refresh: false,
            token_env: "BURN_AUTOMATA_TEST_HF_TOKEN",
        })
        .unwrap();

        assert_eq!(sources.len(), 1);
        assert_eq!(sources[0].slug, "00000005_abcd");
        assert_eq!(sources[0].title.as_deref(), Some("cached preview"));
        assert_eq!(
            sources[0].group.as_deref(),
            Some("omnisvg:mmsvg-illustration")
        );
        assert_eq!(
            sources[0].condition_path,
            paths.root.join("00000005_abcd.png")
        );

        let _ = std::fs::remove_dir_all(cache_dir);
    }

    #[test]
    fn cache_only_resolver_reports_missing_range() {
        let cache_dir = unique_temp_dir("burn_automata_omnisvg_missing");
        let err = resolve_omnisvg_scratch_sources(OmniSvgSourceConfig {
            dataset: OmniSvgDataset::MmsvgIllustration,
            split: "train",
            cache_dir: &cache_dir,
            offset: 0,
            limit: 1,
            page_size: 100,
            download: false,
            refresh: false,
            token_env: "BURN_AUTOMATA_TEST_HF_TOKEN",
        })
        .unwrap_err()
        .to_string();

        assert!(err.contains("OmniSVG cache has 0 usable thumbnails"));
        let _ = std::fs::remove_dir_all(cache_dir);
    }

    #[test]
    fn rows_url_escapes_dataset_id() {
        let url = hf_rows_url("OmniSVG/MMSVG-Illustration", "train", 10, 20);

        assert!(url.contains("dataset=OmniSVG%2FMMSVG-Illustration"));
        assert!(url.contains("split=train"));
        assert!(url.contains("offset=10"));
        assert!(url.contains("length=20"));
    }

    fn unique_temp_dir(prefix: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("{prefix}_{}_{}", std::process::id(), nanos))
    }
}
