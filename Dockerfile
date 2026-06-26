FROM rust:1.93-bookworm AS builder

WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY crates/ crates/

RUN cargo build --release --locked -p vxdb-server

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && rm -rf /var/lib/apt/lists/* \
    && useradd --system --no-create-home --user-group vxdb

COPY --from=builder /app/target/release/vxdb-server /usr/local/bin/vxdb-server

USER vxdb
EXPOSE 8080

ENTRYPOINT ["vxdb-server"]
CMD ["--host", "0.0.0.0", "--port", "8080"]
