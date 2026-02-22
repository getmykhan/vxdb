use serde_json::Value;

use crate::error::{VexError, VexResult};
use crate::types::Metadata;

/// Parsed filter tree that can be evaluated against metadata in-memory
/// or translated to a SQL WHERE clause for the metadata store.
#[derive(Debug, Clone)]
pub enum Filter {
    Eq(String, Value),
    Ne(String, Value),
    Gt(String, Value),
    Gte(String, Value),
    Lt(String, Value),
    Lte(String, Value),
    In(String, Vec<Value>),
    Nin(String, Vec<Value>),
    And(Vec<Filter>),
    Or(Vec<Filter>),
}

impl Filter {
    /// Parse a filter from a JSON value (the dict the user passes).
    /// Format: {"field": {"$op": value}, ...}
    /// Top-level keys are implicitly ANDed.
    /// $and / $or take arrays of sub-filters.
    pub fn parse(value: &Value) -> VexResult<Self> {
        let obj = value
            .as_object()
            .ok_or_else(|| VexError::InvalidFilter("filter must be a JSON object".into()))?;

        let mut conditions = Vec::new();

        for (key, val) in obj {
            if key == "$and" {
                let arr = val
                    .as_array()
                    .ok_or_else(|| VexError::InvalidFilter("$and must be an array".into()))?;
                let subs: VexResult<Vec<Filter>> = arr.iter().map(Filter::parse).collect();
                conditions.push(Filter::And(subs?));
            } else if key == "$or" {
                let arr = val
                    .as_array()
                    .ok_or_else(|| VexError::InvalidFilter("$or must be an array".into()))?;
                let subs: VexResult<Vec<Filter>> = arr.iter().map(Filter::parse).collect();
                conditions.push(Filter::Or(subs?));
            } else {
                // key is a field name
                match val {
                    Value::Object(ops) => {
                        for (op, operand) in ops {
                            conditions.push(parse_operator(key, op, operand)?);
                        }
                    }
                    // Shorthand: {"field": value} means {"field": {"$eq": value}}
                    other => {
                        conditions.push(Filter::Eq(key.clone(), other.clone()));
                    }
                }
            }
        }

        if conditions.len() == 1 {
            Ok(conditions.into_iter().next().unwrap())
        } else {
            Ok(Filter::And(conditions))
        }
    }

    /// Evaluate the filter against an in-memory metadata map.
    pub fn matches(&self, meta: &Metadata) -> bool {
        match self {
            Filter::Eq(field, val) => meta.get(field).map_or(false, |v| v == val),
            Filter::Ne(field, val) => meta.get(field).map_or(true, |v| v != val),
            Filter::Gt(field, val) => meta.get(field).map_or(false, |v| cmp_values(v, val) == Some(std::cmp::Ordering::Greater)),
            Filter::Gte(field, val) => meta.get(field).map_or(false, |v| matches!(cmp_values(v, val), Some(std::cmp::Ordering::Greater | std::cmp::Ordering::Equal))),
            Filter::Lt(field, val) => meta.get(field).map_or(false, |v| cmp_values(v, val) == Some(std::cmp::Ordering::Less)),
            Filter::Lte(field, val) => meta.get(field).map_or(false, |v| matches!(cmp_values(v, val), Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal))),
            Filter::In(field, vals) => meta.get(field).map_or(false, |v| vals.contains(v)),
            Filter::Nin(field, vals) => meta.get(field).map_or(true, |v| !vals.contains(v)),
            Filter::And(subs) => subs.iter().all(|f| f.matches(meta)),
            Filter::Or(subs) => subs.iter().any(|f| f.matches(meta)),
        }
    }

    /// Convert to a SQL WHERE clause for the metadata store.
    /// The metadata is stored as JSON in a TEXT column called `data`.
    pub fn to_sql(&self) -> (String, Vec<Value>) {
        let mut params = Vec::new();
        let clause = self.to_sql_inner(&mut params);
        (clause, params)
    }

    fn to_sql_inner(&self, params: &mut Vec<Value>) -> String {
        match self {
            Filter::Eq(field, val) => {
                params.push(val.clone());
                format!("json_extract(data, '$.{}') = ?{}", field, params.len())
            }
            Filter::Ne(field, val) => {
                params.push(val.clone());
                format!(
                    "(json_extract(data, '$.{}') IS NULL OR json_extract(data, '$.{}') != ?{})",
                    field, field, params.len()
                )
            }
            Filter::Gt(field, val) => {
                params.push(val.clone());
                format!("json_extract(data, '$.{}') > ?{}", field, params.len())
            }
            Filter::Gte(field, val) => {
                params.push(val.clone());
                format!("json_extract(data, '$.{}') >= ?{}", field, params.len())
            }
            Filter::Lt(field, val) => {
                params.push(val.clone());
                format!("json_extract(data, '$.{}') < ?{}", field, params.len())
            }
            Filter::Lte(field, val) => {
                params.push(val.clone());
                format!("json_extract(data, '$.{}') <= ?{}", field, params.len())
            }
            Filter::In(field, vals) => {
                let placeholders: Vec<String> = vals
                    .iter()
                    .map(|v| {
                        params.push(v.clone());
                        format!("?{}", params.len())
                    })
                    .collect();
                format!(
                    "json_extract(data, '$.{}') IN ({})",
                    field,
                    placeholders.join(", ")
                )
            }
            Filter::Nin(field, vals) => {
                let placeholders: Vec<String> = vals
                    .iter()
                    .map(|v| {
                        params.push(v.clone());
                        format!("?{}", params.len())
                    })
                    .collect();
                format!(
                    "(json_extract(data, '$.{}') IS NULL OR json_extract(data, '$.{}') NOT IN ({}))",
                    field,
                    field,
                    placeholders.join(", ")
                )
            }
            Filter::And(subs) => {
                let parts: Vec<String> = subs.iter().map(|f| f.to_sql_inner(params)).collect();
                format!("({})", parts.join(" AND "))
            }
            Filter::Or(subs) => {
                let parts: Vec<String> = subs.iter().map(|f| f.to_sql_inner(params)).collect();
                format!("({})", parts.join(" OR "))
            }
        }
    }
}

fn parse_operator(field: &str, op: &str, operand: &Value) -> VexResult<Filter> {
    match op {
        "$eq" => Ok(Filter::Eq(field.into(), operand.clone())),
        "$ne" => Ok(Filter::Ne(field.into(), operand.clone())),
        "$gt" => Ok(Filter::Gt(field.into(), operand.clone())),
        "$gte" => Ok(Filter::Gte(field.into(), operand.clone())),
        "$lt" => Ok(Filter::Lt(field.into(), operand.clone())),
        "$lte" => Ok(Filter::Lte(field.into(), operand.clone())),
        "$in" => {
            let arr = operand
                .as_array()
                .ok_or_else(|| VexError::InvalidFilter("$in requires an array".into()))?;
            Ok(Filter::In(field.into(), arr.clone()))
        }
        "$nin" => {
            let arr = operand
                .as_array()
                .ok_or_else(|| VexError::InvalidFilter("$nin requires an array".into()))?;
            Ok(Filter::Nin(field.into(), arr.clone()))
        }
        _ => Err(VexError::InvalidFilter(format!("unknown operator: {}", op))),
    }
}

fn cmp_values(a: &Value, b: &Value) -> Option<std::cmp::Ordering> {
    match (a, b) {
        (Value::Number(a), Value::Number(b)) => {
            let af = a.as_f64()?;
            let bf = b.as_f64()?;
            af.partial_cmp(&bf)
        }
        (Value::String(a), Value::String(b)) => Some(a.cmp(b)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::collections::HashMap;

    fn meta(pairs: &[(&str, Value)]) -> Metadata {
        pairs.iter().map(|(k, v)| (k.to_string(), v.clone())).collect()
    }

    #[test]
    fn test_parse_eq() {
        let f = Filter::parse(&json!({"color": {"$eq": "red"}})).unwrap();
        assert!(f.matches(&meta(&[("color", json!("red"))])));
        assert!(!f.matches(&meta(&[("color", json!("blue"))])));
    }

    #[test]
    fn test_parse_shorthand_eq() {
        let f = Filter::parse(&json!({"color": "red"})).unwrap();
        assert!(f.matches(&meta(&[("color", json!("red"))])));
    }

    #[test]
    fn test_parse_ne() {
        let f = Filter::parse(&json!({"color": {"$ne": "red"}})).unwrap();
        assert!(!f.matches(&meta(&[("color", json!("red"))])));
        assert!(f.matches(&meta(&[("color", json!("blue"))])));
        assert!(f.matches(&HashMap::new())); // missing field
    }

    #[test]
    fn test_parse_gt_gte_lt_lte() {
        let f_gt = Filter::parse(&json!({"price": {"$gt": 10}})).unwrap();
        let f_gte = Filter::parse(&json!({"price": {"$gte": 10}})).unwrap();
        let f_lt = Filter::parse(&json!({"price": {"$lt": 10}})).unwrap();
        let f_lte = Filter::parse(&json!({"price": {"$lte": 10}})).unwrap();

        let m5 = meta(&[("price", json!(5))]);
        let m10 = meta(&[("price", json!(10))]);
        let m15 = meta(&[("price", json!(15))]);

        assert!(!f_gt.matches(&m5));
        assert!(!f_gt.matches(&m10));
        assert!(f_gt.matches(&m15));

        assert!(!f_gte.matches(&m5));
        assert!(f_gte.matches(&m10));
        assert!(f_gte.matches(&m15));

        assert!(f_lt.matches(&m5));
        assert!(!f_lt.matches(&m10));
        assert!(!f_lt.matches(&m15));

        assert!(f_lte.matches(&m5));
        assert!(f_lte.matches(&m10));
        assert!(!f_lte.matches(&m15));
    }

    #[test]
    fn test_parse_in() {
        let f = Filter::parse(&json!({"color": {"$in": ["red", "blue"]}})).unwrap();
        assert!(f.matches(&meta(&[("color", json!("red"))])));
        assert!(f.matches(&meta(&[("color", json!("blue"))])));
        assert!(!f.matches(&meta(&[("color", json!("green"))])));
    }

    #[test]
    fn test_parse_nin() {
        let f = Filter::parse(&json!({"color": {"$nin": ["red", "blue"]}})).unwrap();
        assert!(!f.matches(&meta(&[("color", json!("red"))])));
        assert!(!f.matches(&meta(&[("color", json!("blue"))])));
        assert!(f.matches(&meta(&[("color", json!("green"))])));
        assert!(f.matches(&HashMap::new())); // missing field
    }

    #[test]
    fn test_implicit_and() {
        let f = Filter::parse(&json!({
            "color": {"$eq": "red"},
            "price": {"$lt": 100}
        }))
        .unwrap();

        assert!(f.matches(&meta(&[("color", json!("red")), ("price", json!(50))])));
        assert!(!f.matches(&meta(&[("color", json!("blue")), ("price", json!(50))])));
        assert!(!f.matches(&meta(&[("color", json!("red")), ("price", json!(150))])));
    }

    #[test]
    fn test_explicit_and() {
        let f = Filter::parse(&json!({
            "$and": [
                {"color": {"$eq": "red"}},
                {"price": {"$gt": 10}}
            ]
        }))
        .unwrap();

        assert!(f.matches(&meta(&[("color", json!("red")), ("price", json!(20))])));
        assert!(!f.matches(&meta(&[("color", json!("red")), ("price", json!(5))])));
    }

    #[test]
    fn test_or() {
        let f = Filter::parse(&json!({
            "$or": [
                {"color": "red"},
                {"color": "blue"}
            ]
        }))
        .unwrap();

        assert!(f.matches(&meta(&[("color", json!("red"))])));
        assert!(f.matches(&meta(&[("color", json!("blue"))])));
        assert!(!f.matches(&meta(&[("color", json!("green"))])));
    }

    #[test]
    fn test_nested_and_or() {
        let f = Filter::parse(&json!({
            "$and": [
                {"$or": [
                    {"color": "red"},
                    {"color": "blue"}
                ]},
                {"price": {"$lte": 100}}
            ]
        }))
        .unwrap();

        assert!(f.matches(&meta(&[("color", json!("red")), ("price", json!(50))])));
        assert!(f.matches(&meta(&[("color", json!("blue")), ("price", json!(100))])));
        assert!(!f.matches(&meta(&[("color", json!("green")), ("price", json!(50))])));
        assert!(!f.matches(&meta(&[("color", json!("red")), ("price", json!(200))])));
    }

    #[test]
    fn test_invalid_filter() {
        assert!(Filter::parse(&json!("not an object")).is_err());
        assert!(Filter::parse(&json!({"f": {"$unknown": 1}})).is_err());
        assert!(Filter::parse(&json!({"$in": "not array"})).is_ok()); // $in at top level is a field name
        assert!(Filter::parse(&json!({"f": {"$in": "not array"}})).is_err());
    }

    #[test]
    fn test_to_sql_eq() {
        let f = Filter::parse(&json!({"color": {"$eq": "red"}})).unwrap();
        let (sql, params) = f.to_sql();
        assert_eq!(sql, "json_extract(data, '$.color') = ?1");
        assert_eq!(params, vec![json!("red")]);
    }

    #[test]
    fn test_to_sql_and() {
        let f = Filter::parse(&json!({
            "$and": [
                {"color": {"$eq": "red"}},
                {"price": {"$gt": 10}}
            ]
        }))
        .unwrap();
        let (sql, params) = f.to_sql();
        assert!(sql.contains("AND"));
        assert_eq!(params.len(), 2);
    }

    #[test]
    fn test_to_sql_in() {
        let f = Filter::parse(&json!({"color": {"$in": ["red", "blue"]}})).unwrap();
        let (sql, params) = f.to_sql();
        assert!(sql.contains("IN"));
        assert_eq!(params.len(), 2);
    }

    #[test]
    fn test_string_comparison() {
        let f = Filter::parse(&json!({"name": {"$gt": "b"}})).unwrap();
        assert!(f.matches(&meta(&[("name", json!("c"))])));
        assert!(!f.matches(&meta(&[("name", json!("a"))])));
    }
}
