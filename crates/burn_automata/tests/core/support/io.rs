use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
};

pub(crate) fn temp_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("burn_automata_{}_{}", std::process::id(), name))
}

pub(crate) fn workspace_path(relative: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(relative)
}

pub(crate) fn write_fake_pytorch_checkpoint(path: &Path) {
    let file = fs::File::create(path).unwrap();
    let mut zip = zip::ZipWriter::new(file);
    let options =
        zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);

    zip.start_file("fake/byteorder", options).unwrap();
    zip.write_all(b"little").unwrap();
    zip.start_file("fake/data/0", options).unwrap();
    write_f32s(&mut zip, &[0.1]);
    zip.start_file("fake/data/1", options).unwrap();
    write_f32s(&mut zip, &[0.5]);
    zip.start_file("fake/data/2", options).unwrap();
    write_f32s(&mut zip, &[0.01; 12]);
    zip.start_file("fake/data/3", options).unwrap();
    write_f32s(&mut zip, &[0.0; 2]);
    zip.start_file("fake/data/4", options).unwrap();
    write_f32s(&mut zip, &[0.02; 6]);
    zip.finish().unwrap();
}

fn write_f32s<W: Write>(writer: &mut W, values: &[f32]) {
    for value in values {
        writer.write_all(&value.to_le_bytes()).unwrap();
    }
}
