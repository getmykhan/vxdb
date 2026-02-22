use thiserror::Error;

#[derive(Error, Debug)]
pub enum VexError {
    #[error("collection '{0}' not found")]
    CollectionNotFound(String),

    #[error("collection '{0}' already exists")]
    CollectionAlreadyExists(String),

    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("document '{0}' not found")]
    DocumentNotFound(String),

    #[error("empty vector provided")]
    EmptyVector,

    #[error("invalid filter: {0}")]
    InvalidFilter(String),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("{0}")]
    Internal(String),
}

pub type VexResult<T> = Result<T, VexError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let e = VexError::CollectionNotFound("test".into());
        assert_eq!(e.to_string(), "collection 'test' not found");
    }

    #[test]
    fn test_dimension_mismatch_display() {
        let e = VexError::DimensionMismatch { expected: 384, got: 128 };
        assert_eq!(e.to_string(), "dimension mismatch: expected 384, got 128");
    }

    #[test]
    fn test_result_type() {
        let ok: VexResult<i32> = Ok(42);
        assert_eq!(ok.unwrap(), 42);

        let err: VexResult<i32> = Err(VexError::EmptyVector);
        assert!(err.is_err());
    }
}
