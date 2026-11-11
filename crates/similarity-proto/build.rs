//! Generates the gRPC client and server from `proto/`.
//!
//! Requires `protoc` on the `PATH` (`brew install protobuf`, `apt-get install protobuf-compiler`).

use std::path::PathBuf;

const PROTO: &str = "proto/recordsimilarity/v1/similarity.proto";

fn main() -> Result<(), Box<dyn std::error::Error>> {
  println!("cargo:rerun-if-changed={PROTO}");
  let out_dir = PathBuf::from(std::env::var("OUT_DIR")?);
  tonic_prost_build::configure()
    .build_server(true)
    .build_client(true)
    // The descriptor set backs gRPC server reflection, so `grpcurl` and friends work without the .proto file.
    .file_descriptor_set_path(out_dir.join("recordsimilarity_descriptor.bin"))
    .compile_protos(&[PROTO], &["proto"])?;
  Ok(())
}
