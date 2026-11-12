# syntax=docker/dockerfile:1

# ---- build -------------------------------------------------------------------------------------------------------
# Debian 13 for every stage. The server links a prebuilt ONNX Runtime library that is compiled against the C++
# standard library of GCC 13, and Debian 12 has GCC 12's: the build fails at the link step there.
ARG BUILD_IMAGE=rust:1.96-trixie
ARG EXPORT_IMAGE=ghcr.io/astral-sh/uv:0.11.28-python3.13-trixie-slim
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
ENV RECORD_SIMILARITY_COMMIT=${RECORD_SIMILARITY_COMMIT}
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/src/target \
    cargo build --release --locked -p similarity-server \
  && cp target/release/record-similarity /usr/local/bin/record-similarity

# The model is baked into the image, checksum-verified at build time, so that a running container never needs the
# network to start and every replica is guaranteed to run the same weights.
RUN record-similarity model fetch --dir /opt/models/mdbr-leaf-mt

# ---- export ------------------------------------------------------------------------------------------------------
# The file ONNX Runtime runs the model from, written by PyTorch from the weights fetched above. The script checks
# it against PyTorch before keeping it, and the server checks it against the weights again every time it starts.
# Python is in this stage and nowhere else: only the model directory is taken from it.
FROM ${EXPORT_IMAGE} AS export

COPY --from=build /opt/models /opt/models
COPY tools/export_onnx.py /src/tools/export_onnx.py
# The CPU builds of PyTorch. Without these two, an x86-64 build resolves to the CUDA builds and downloads several
# gigabytes it never uses. `unsafe-best-match` only means "look at both indexes for every package".
ENV UV_INDEX=https://download.pytorch.org/whl/cpu \
    UV_INDEX_STRATEGY=unsafe-best-match
RUN --mount=type=cache,target=/root/.cache/uv \
    uv run /src/tools/export_onnx.py /opt/models/mdbr-leaf-mt

# ---- runtime -----------------------------------------------------------------------------------------------------
FROM ${RUNTIME_IMAGE}

# Root certificates, for TLS to a managed Qdrant.
RUN apt-get update \
  && apt-get install -y --no-install-recommends ca-certificates \
  && rm -rf /var/lib/apt/lists/* \
  && useradd --system --uid 1001 --no-create-home similarity

COPY --from=build /usr/local/bin/record-similarity /usr/local/bin/record-similarity
COPY --from=export /opt/models /opt/models
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
