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
const HTTP_FETCH_MAX_BACKOFF: Duration = Duration::from_secs(30);

#[derive(Clone, Copy, Debug)]
pub(in crate::hyper::e2e_rollout) struct OmniSvgSourceConfig<'a> {
    pub(in crate::hyper::e2e_rollout) dataset: OmniSvgDataset,
    pub(in crate::hyper::e2e_rollout) split: &'a str,
    pub(in crate::hyper::e2e_rollout) cache_dir: &'a Path,
    pub(in crate::hyper::e2e_rollout) offset: usize,
    pub(in crate::hyper::e2e_rollout) limit: usize,
    pub(in crate::hyper::e2e_rollout) page_size: usize,
    pub(in crate::hyper::e2e_rollout) download: bool,
    pub(in crate::hyper::e2e_rollout) refresh: bool,
    pub(in crate::hyper::e2e_rollout) token_env: &'a str,
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
            let response_text = fetch_hf_rows_page(
                config.dataset.dataset_id(),
                config.split,
                page_offset,
                page_len,
                token.as_deref(),
            )?;
            let rows = parse_omnisvg_rows_response(&response_text, page_offset)?;
            for row in rows {
                if row.source_offset >= requested_end {
                    continue;
                }
                let thumbnail_file = thumbnail_file_name(row.source_offset, &row.id, &row.image);
                let thumbnail_path = paths.root.join(&thumbnail_file);
                if config.refresh || !thumbnail_path.exists() {
                    let bytes = image_bytes(row.image.clone(), token.as_deref())?;
                    if bytes.is_empty() {
                        return Err(std::io::Error::other(format!(
                            "OmniSVG thumbnail for row {} was empty",
                            row.source_offset
                        ))
                        .into());
                    }
                    std::fs::write(&thumbnail_path, bytes)?;
                }
                entries_by_offset.insert(
                    row.source_offset,
                    OmniSvgCacheEntry {
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
                    },
                );
            }
            save_entries_manifest(paths, config.dataset, config.split, &entries_by_offset)?;
        }
        page_offset += page_len;
    }

    let manifest = entries_manifest(config.dataset, config.split, &entries_by_offset);
    save_manifest(paths, &manifest)?;
    Ok(manifest)
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
    std::fs::write(&paths.manifest, serde_json::to_string_pretty(manifest)?)?;
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
    http_get_text(&url, token)
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

fn http_get_text(url: &str, token: Option<&str>) -> Result<String, Box<dyn std::error::Error>> {
    let bytes = http_get_bytes(url, token)?;
    Ok(String::from_utf8(bytes)?)
}

fn http_get_bytes(url: &str, token: Option<&str>) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let mut delay = Duration::from_millis(500);
    for attempt in 1..=HTTP_FETCH_ATTEMPTS {
        match http_get_bytes_once(url, token) {
            Ok(bytes) => return Ok(bytes),
            Err(err) if attempt == HTTP_FETCH_ATTEMPTS => return Err(err),
            Err(err) => {
                eprintln!(
                    "warning: HTTP fetch attempt {attempt}/{HTTP_FETCH_ATTEMPTS} failed for {url}: {err}; retrying in {} ms",
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
        .set("User-Agent", HTTP_USER_AGENT)
        .set("Accept", "*/*");
    let bearer;
    if let Some(token) = token {
        bearer = format!("Bearer {token}");
        request = request.set("Authorization", &bearer);
    }
    let response = request
        .call()
        .map_err(|err| std::io::Error::other(format!("HTTP fetch failed for {url}: {err}")))?;
    let mut bytes = Vec::new();
    response
        .into_reader()
        .read_to_end(&mut bytes)
        .map_err(|err| std::io::Error::other(format!("HTTP body read failed for {url}: {err}")))?;
    Ok(bytes)
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
