//! Downloading ONNX Runtime's library: Microsoft's own build of it, taken out of its `NuGet` package.
//!
//! The forward pass is computed by a C++ library that is not part of this binary (docs/decisions.md, D34): it is a
//! file beside it, opened when a model is loaded. This module is the one place that says which file. It is
//! Microsoft's release build of one exact version, downloaded from nuget.org, where Microsoft publishes the
//! native library for every platform in one signed package, and it is used only if the package and the library
//! taken out of it both have the SHA-256 compiled in here. A library is 25 to 45 MB of machine code nobody can
//! review by eye, so, as with the model (see [`crate::fetch`]), "what we tested is what we run" is checked by a
//! machine.
//!
//! It runs only when a human or an image build asks for it (`record-similarity runtime fetch`). The server never
//! downloads anything: it opens the path it is given.

use std::io::{Read, Seek, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, bail};
use sha2::{Digest, Sha256};

use crate::fetch::{download, hex, sha256_of_file};

/// The version of ONNX Runtime that is fetched: the one the engine was measured with.
pub const VERSION: &str = "1.31.1";
/// Where `runtime fetch` puts the library, and where the `just` recipes look for it.
pub const DEFAULT_DIR: &str = "onnxruntime";
/// SHA-256 of `Microsoft.ML.OnnxRuntime` [`VERSION`] as nuget.org serves it.
const PACKAGE_SHA256: &str = "6871ccbc97f3d5bdb4095a1dcec33aa78d56f6ed8de5980ef5fa1d9f2e7c679a";
/// What is kept beside the library because its license asks for it wherever the library goes.
const NOTICES: [&str; 2] = ["LICENSE", "ThirdPartyNotices.txt"];

/// One platform's library: where it is inside the package, and what it has to be.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Platform {
  /// `std::env::consts::OS` of the platform.
  pub os: &'static str,
  /// `std::env::consts::ARCH` of the platform.
  pub arch: &'static str,
  /// The library's path inside the package.
  pub entry: &'static str,
  /// SHA-256 of the library.
  pub sha256: &'static str,
}

impl Platform {
  /// The library's file name, which is the one the loader looks for.
  pub fn file_name(&self) -> &'static str {
    self.entry.rsplit('/').next().unwrap_or(self.entry)
  }
}

/// The platforms the server is built for, with the library of [`VERSION`] for each.
pub const PLATFORMS: [Platform; 3] = [
  Platform {
    os: "linux",
    arch: "x86_64",
    entry: "runtimes/linux-x64/native/libonnxruntime.so",
    sha256: "a96f132e5eb5ca04e23a7d2fc1078fb95aa591e8d71d0f237702ece3a1e0f4ea",
  },
  Platform {
    os: "linux",
    arch: "aarch64",
    entry: "runtimes/linux-arm64/native/libonnxruntime.so",
    sha256: "7e17c55d675be73f67a218dbecb55522395f61dafb3da03ebc7ade56a4c2f813",
  },
  Platform {
    os: "macos",
    arch: "aarch64",
    entry: "runtimes/osx-arm64/native/libonnxruntime.dylib",
    sha256: "2685dc296147c47b0dbd8e0615547b12fc0c445d158d1f374cbd0705d9626fdc",
  },
];

/// The library for an operating system and an architecture, named as `std::env::consts` names them.
///
/// # Errors
///
/// Returns an error that lists the supported platforms if this is not one of them.
pub fn platform(os: &str, arch: &str) -> anyhow::Result<&'static Platform> {
  PLATFORMS
    .iter()
    .find(|platform| platform.os == os && platform.arch == arch)
    .with_context(|| {
      let known: Vec<String> = PLATFORMS
        .iter()
        .map(|platform| format!("{} {}", platform.os, platform.arch))
        .collect();
      format!(
        "no ONNX Runtime library is pinned for {os} {arch}; the ones that are: {}",
        known.join(", ")
      )
    })
}

/// Where the package is downloaded from.
pub fn package_url() -> String {
  format!(
    "https://api.nuget.org/v3-flatcontainer/microsoft.ml.onnxruntime/{VERSION}/microsoft.ml.onnxruntime.{VERSION}.nupkg"
  )
}

/// What a fetch left behind.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Fetched {
  /// The library, in place.
  pub library: PathBuf,
  /// Its SHA-256, which is the pinned one.
  pub sha256: String,
  /// Whether it was already there with the pinned digest, so that nothing was downloaded.
  pub cached: bool,
}

/// Puts this platform's library of [`VERSION`] into `dir`, with the notices its license asks to travel with it.
///
/// A library already there with the pinned digest is left alone and nothing is downloaded. Otherwise the package
/// is downloaded into `dir`, checked, the library taken out of it and checked, and the package removed.
///
/// # Errors
///
/// Returns an error if this platform has no pinned library, if the download fails, or if the package or the
/// library is not what was pinned, in which case neither is left in `dir`.
pub fn fetch(dir: &Path) -> anyhow::Result<Fetched> {
  let platform = platform(std::env::consts::OS, std::env::consts::ARCH)?;
  let library = dir.join(platform.file_name());
  if library.is_file() && sha256_of_file(&library)? == platform.sha256 {
    return Ok(Fetched {
      library,
      sha256: platform.sha256.to_owned(),
      cached: true,
    });
  }

  let package = dir.join(format!("onnxruntime-{VERSION}.nupkg"));
  let found = download(&package_url(), &package)?;
  if found != PACKAGE_SHA256 {
    // Best effort: the mismatch is the error worth reporting, not a failure to clean up after it.
    let _ = std::fs::remove_file(&package);
    bail!(
      "checksum mismatch for the ONNX Runtime {VERSION} package: expected {PACKAGE_SHA256}, got {found}; the file \
       was removed"
    );
  }
  let archive = std::fs::File::open(&package).with_context(|| format!("cannot open {}", package.display()))?;
  let taken = take(archive, platform, dir);
  let _ = std::fs::remove_file(&package);
  Ok(Fetched {
    library,
    sha256: taken?,
    cached: false,
  })
}

/// Takes `platform`'s library out of a package into `dir`, under its own file name, and returns its SHA-256. The
/// notices are taken with it when the package has them.
///
/// # Errors
///
/// Returns an error if `package` is not a zip archive, has no library for the platform, or has one that is not
/// the pinned one, in which case nothing is left under the library's name.
pub fn take(package: impl Read + Seek, platform: &Platform, dir: &Path) -> anyhow::Result<String> {
  let mut archive = zip::ZipArchive::new(package).context("the ONNX Runtime package is not a zip archive")?;
  let library = dir.join(platform.file_name());
  let sha256 = {
    let entry = archive
      .by_name(platform.entry)
      .with_context(|| format!("the ONNX Runtime package has no `{}`", platform.entry))?;
    write_whole(entry, &library)?
  };
  if sha256 != platform.sha256 {
    let _ = std::fs::remove_file(&library);
    bail!(
      "checksum mismatch for {}: expected {}, got {sha256}; the file was removed",
      platform.file_name(),
      platform.sha256
    );
  }
  for notice in NOTICES {
    // A package is whole or it is refused above; a notice it does not carry is not a reason to refuse a library.
    if let Ok(entry) = archive.by_name(notice) {
      write_whole(entry, &dir.join(notice))?;
    }
  }
  Ok(sha256)
}

/// Streams `from` into `destination` through a file beside it, so that nothing truncated is ever under the real
/// name, and returns the SHA-256 of what was written.
fn write_whole(mut from: impl Read, destination: &Path) -> anyhow::Result<String> {
  if let Some(parent) = destination.parent() {
    std::fs::create_dir_all(parent).with_context(|| format!("cannot create {}", parent.display()))?;
  }
  let partial = destination.with_extension("part");
  let mut file = std::fs::File::create(&partial).with_context(|| format!("cannot create {}", partial.display()))?;
  let mut hasher = Sha256::new();
  let mut buffer = vec![0_u8; 1 << 16];
  loop {
    let read = from
      .read(&mut buffer)
      .with_context(|| format!("cannot read `{}` out of the package", destination.display()))?;
    let Some(chunk) = buffer.get(..read).filter(|chunk| !chunk.is_empty()) else {
      break;
    };
    hasher.update(chunk);
    file
      .write_all(chunk)
      .with_context(|| format!("cannot write {}", partial.display()))?;
  }
  file
    .sync_all()
    .with_context(|| format!("cannot flush {}", partial.display()))?;
  std::fs::rename(&partial, destination).with_context(|| format!("cannot move {} into place", partial.display()))?;
  Ok(hex(&hasher.finalize()))
}

#[cfg(test)]
mod tests {
  use std::io::Cursor;

  use zip::write::SimpleFileOptions;

  use super::*;

  const LIBRARY: &[u8] = b"not really a library, but bytes with a digest";

  /// A package holding `files`, as a `NuGet` package holds them.
  fn package(files: &[(&str, &[u8])]) -> Cursor<Vec<u8>> {
    let mut writer = zip::ZipWriter::new(Cursor::new(Vec::new()));
    for (name, bytes) in files {
      writer.start_file(*name, SimpleFileOptions::default()).unwrap();
      writer.write_all(bytes).unwrap();
    }
    Cursor::new(writer.finish().unwrap().into_inner())
  }

  /// A platform whose pinned library is `bytes`. The digest is leaked: a `Platform` holds `'static` text.
  fn pinned(bytes: &[u8]) -> Platform {
    Platform {
      os: "test",
      arch: "test",
      entry: "runtimes/test/native/libonnxruntime.so",
      sha256: Box::leak(hex(&Sha256::digest(bytes)).into_boxed_str()),
    }
  }

  #[test]
  fn the_library_is_taken_out_of_the_package_under_its_own_name_with_its_notices() {
    let dir = tempfile::tempdir().unwrap();
    let platform = pinned(LIBRARY);
    let archive = package(&[
      ("runtimes/other/native/libonnxruntime.so", b"another platform's"),
      (platform.entry, LIBRARY),
      ("LICENSE", b"MIT"),
      ("ThirdPartyNotices.txt", b"notices"),
      ("lib/net8.0/Microsoft.ML.OnnxRuntime.dll", b"not wanted"),
    ]);

    let sha256 = take(archive, &platform, dir.path()).unwrap();

    assert_eq!(sha256, platform.sha256);
    assert_eq!(std::fs::read(dir.path().join("libonnxruntime.so")).unwrap(), LIBRARY);
    assert_eq!(std::fs::read(dir.path().join("LICENSE")).unwrap(), b"MIT");
    let mut left: Vec<String> = std::fs::read_dir(dir.path())
      .unwrap()
      .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
      .collect();
    left.sort();
    assert_eq!(left, ["LICENSE", "ThirdPartyNotices.txt", "libonnxruntime.so"]);
  }

  #[test]
  fn a_library_that_is_not_the_pinned_one_is_refused_and_nothing_is_left() {
    let dir = tempfile::tempdir().unwrap();
    let platform = pinned(LIBRARY);
    let archive = package(&[(platform.entry, b"something else entirely"), ("LICENSE", b"MIT")]);

    let error = take(archive, &platform, dir.path()).unwrap_err();

    assert!(error.to_string().contains("checksum mismatch"), "{error}");
    assert_eq!(
      std::fs::read_dir(dir.path()).unwrap().count(),
      0,
      "nothing may be left behind"
    );
  }

  #[test]
  fn a_package_without_this_platforms_library_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let platform = pinned(LIBRARY);
    let archive = package(&[("runtimes/other/native/libonnxruntime.so", LIBRARY)]);

    let error = take(archive, &platform, dir.path()).unwrap_err();

    assert!(error.to_string().contains(platform.entry), "{error}");
    assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 0);
  }

  #[test]
  fn a_file_that_is_not_a_package_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let error = take(
      Cursor::new(b"<html>not found</html>".to_vec()),
      &pinned(LIBRARY),
      dir.path(),
    )
    .unwrap_err();
    assert!(error.to_string().contains("not a zip archive"), "{error}");
  }

  #[test]
  fn a_library_without_its_notices_is_still_taken() {
    let dir = tempfile::tempdir().unwrap();
    let platform = pinned(LIBRARY);
    take(package(&[(platform.entry, LIBRARY)]), &platform, dir.path()).unwrap();
    assert!(dir.path().join("libonnxruntime.so").is_file());
  }

  #[test]
  fn every_platform_has_a_library_and_a_whole_checksum() {
    for platform in &PLATFORMS {
      assert!(platform.entry.starts_with("runtimes/"), "{platform:?}");
      assert!(platform.file_name().starts_with("libonnxruntime."), "{platform:?}");
      assert_eq!(platform.sha256.len(), 64, "{platform:?}");
      assert!(platform.sha256.chars().all(|c| c.is_ascii_hexdigit()), "{platform:?}");
    }
    assert_eq!(PACKAGE_SHA256.len(), 64);
    // Two platforms with one digest would be a copying mistake.
    let mut digests: Vec<&str> = PLATFORMS.iter().map(|platform| platform.sha256).collect();
    digests.sort_unstable();
    digests.dedup();
    assert_eq!(digests.len(), PLATFORMS.len());
  }

  #[test]
  fn the_library_is_named_as_the_loader_looks_for_it() {
    // What `runtime fetch` writes has to be the file the engine opens when it is told nothing: this platform's.
    let platform = platform(std::env::consts::OS, std::env::consts::ARCH).unwrap();
    assert_eq!(platform.file_name(), similarity_core::model::onnx::library_file_name());
  }

  #[test]
  fn a_platform_without_a_library_is_refused_with_the_ones_that_have_one() {
    let error = platform("windows", "x86_64").unwrap_err().to_string();
    assert!(error.contains("windows x86_64"), "{error}");
    assert!(error.contains("linux x86_64"), "{error}");
    assert!(error.contains("macos aarch64"), "{error}");
  }

  #[test]
  fn the_package_is_the_pinned_version_from_nuget() {
    let url = package_url();
    assert!(url.starts_with("https://api.nuget.org/"), "{url}");
    assert!(
      url.ends_with(&format!("microsoft.ml.onnxruntime.{VERSION}.nupkg")),
      "{url}"
    );
  }
}
