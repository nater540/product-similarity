# syntax=docker/dockerfile:1

# ---- build -------------------------------------------------------------------------------------------------------
# Debian 13 for both stages. The default build would run on Debian 12 as well; the ONNX build
# (`--build-arg FEATURES=onnx`) would not link there, because the prebuilt ONNX Runtime library is compiled against
# the C++ standard library of GCC 13 and Debian 12 has GCC 12's. One base for both keeps the two images alike.
ARG BUILD_IMAGE=rust:1.96-trixie
ARG RUNTIME_IMAGE=debian:trixie-slim

FROM ${BUILD_IMAGE} AS build

# `protoc` compiles the gRPC definitions (crates/similarity-proto/build.rs).
RUN apt-get update \
  && apt-get install -y --no-install-recommends protobuf-compiler \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . .
# The build has no `.git` to ask (it is in `.dockerignore`), so the commit the image is built from is passed in:
# `docker build --build-arg RECORD_SIMILARITY_COMMIT=$(git rev-parse --short=12 HEAD) .`, which is what `just image`
# does. It ends up in the `provenance.json` of the model fetched below.
ARG RECORD_SIMILARITY_COMMIT=unknown
# Cargo features of the server, empty for the default build. `--build-arg FEATURES=onnx` links ONNX Runtime in,
# for `engine = "onnx"`; the model directory then also needs `onnx/model.onnx` (`just export-onnx`), which this
# image does not make.
ARG FEATURES=""
ENV RECORD_SIMILARITY_COMMIT=${RECORD_SIMILARITY_COMMIT}
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/src/target \
    cargo build --release --locked -p similarity-server ${FEATURES:+--features ${FEATURES}} \
  && cp target/release/record-similarity /usr/local/bin/record-similarity

# The model is baked into the image, checksum-verified at build time, so that a running container never needs the
# network to start and every replica is guaranteed to run the same weights.
RUN record-similarity model fetch --dir /opt/models/mdbr-leaf-mt

# ---- runtime -----------------------------------------------------------------------------------------------------
FROM ${RUNTIME_IMAGE}

# Root certificates, for TLS to a managed Qdrant.
RUN apt-get update \
  && apt-get install -y --no-install-recommends ca-certificates \
  && rm -rf /var/lib/apt/lists/* \
  && useradd --system --uid 1001 --no-create-home similarity

COPY --from=build /usr/local/bin/record-similarity /usr/local/bin/record-similarity
COPY --from=build /opt/models /opt/models
COPY config.example.toml /etc/record-similarity/config.toml
# The example config uses paths and addresses that suit a checkout; a container listens on every interface and
# reads the baked-in model.
RUN sed -i \
    -e 's|^listen = .*|listen = "0.0.0.0:50051"|' \
    -e 's|^dir = .*|dir = "/opt/models/mdbr-leaf-mt"|' \
    -e 's|^url = .*|url = "http://qdrant:6334"|' \
    /etc/record-similarity/config.toml

USER similarity
EXPOSE 50051
ENTRYPOINT ["record-similarity"]
CMD ["serve", "--config", "/etc/record-similarity/config.toml"]
