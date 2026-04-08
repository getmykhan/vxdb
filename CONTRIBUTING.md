# Contributing to vxdb

Thanks for your interest in contributing to vxdb! This guide will help you get set up.

## Development Setup

### Prerequisites

- [Rust](https://rustup.rs/) (stable toolchain)
- Python 3.9+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Getting Started

```bash
git clone https://github.com/getmykhan/vxdb.git && cd vxdb

# Rust: build and test
cargo build --all
cargo test --all

# Python: set up virtual environment and build
uv venv .venv && source .venv/bin/activate
uv pip install maturin pytest httpx ruff
maturin develop
PYTHONPATH=python pytest tests/ -v
```

### Project Structure

```
vxdb/
├── crates/
│   ├── vxdb-core/       # Rust engine: indexes, distance, storage, hybrid search
│   ├── vxdb-python/     # PyO3 bindings
│   └── vxdb-server/     # Axum REST API server
├── python/vxdb/         # Python package (client SDK, embedding interface)
├── examples/            # Jupyter notebooks
└── tests/               # Python integration tests
```

## Making Changes

1. **Fork the repo** and create a branch from `main`.
2. **Write tests** for any new functionality.
3. **Run the full test suite** before submitting:
   ```bash
   cargo test --all
   cargo clippy --all -- -D warnings
   PYTHONPATH=python pytest tests/ -v
   ruff check python/ tests/
   ruff format --check python/ tests/
   ```
4. **Submit a pull request** with a clear description of the change.

## Code Style

- **Rust**: Follow standard `rustfmt` formatting. Run `cargo fmt --all` before committing.
- **Python**: We use [ruff](https://docs.astral.sh/ruff/) for linting and formatting. Run `ruff check --fix` and `ruff format` before committing.

## Pull Request Guidelines

- Keep PRs focused on a single change.
- Include tests for new features or bug fixes.
- Update documentation if the public API changes.
- Ensure CI passes before requesting review.

## Reporting Issues

Use [GitHub Issues](https://github.com/getmykhan/vxdb/issues) to report bugs or request features. Include:

- Steps to reproduce (for bugs)
- Expected vs. actual behavior
- Python version, OS, and vxdb version

## License

By contributing, you agree that your contributions will be licensed under the [Apache 2.0 License](LICENSE).
