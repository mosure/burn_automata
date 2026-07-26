use js_sys::Uint8Array;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::JsFuture;

pub(super) async fn fetch_bytes(url: &str) -> Result<Vec<u8>, String> {
    let window = web_sys::window().ok_or_else(|| "browser window is unavailable".to_string())?;
    let response = JsFuture::from(window.fetch_with_str(url))
        .await
        .map_err(|error| format!("failed to fetch {url}: {error:?}"))?
        .dyn_into::<web_sys::Response>()
        .map_err(|_| format!("fetch for {url} returned a non-response value"))?;
    if !response.ok() {
        return Err(format!(
            "failed to fetch {url}: HTTP {} {}",
            response.status(),
            response.status_text(),
        ));
    }
    let buffer = response
        .array_buffer()
        .map_err(|error| format!("failed to read {url} response: {error:?}"))?;
    let buffer = JsFuture::from(buffer)
        .await
        .map_err(|error| format!("failed to download {url}: {error:?}"))?;
    Ok(Uint8Array::new(&buffer).to_vec())
}
