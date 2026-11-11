//! The wire contract of the record-similarity service.
//!
//! The `.proto` file under `proto/` is the source of truth and carries the documentation; this crate only compiles
//! it. Both the server and any Rust client depend on this crate, so neither can drift from the other.

/// Version 1 of the API (`package recordsimilarity.v1`).
#[allow(
  missing_docs,
  clippy::all,
  clippy::pedantic,
  clippy::arithmetic_side_effects,
  clippy::indexing_slicing,
  reason = "generated code"
)]
pub mod v1 {
  include!(concat!(env!("OUT_DIR"), "/recordsimilarity.v1.rs"));
}

/// The encoded `FileDescriptorSet` for [`v1`], for gRPC server reflection.
pub const FILE_DESCRIPTOR_SET: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/recordsimilarity_descriptor.bin"));
