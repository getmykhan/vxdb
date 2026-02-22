FROM rust:1.83-bookworm AS builder

WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY crates/ crates/

RUN cargo build --release -p vxdb-server

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/vxdb-server /usr/local/bin/vxdb-server

EXPOSE 8080

ENTRYPOINT ["vxdb-server"]
CMD ["--host", "0.0.0.0", "--port", "8080"]
