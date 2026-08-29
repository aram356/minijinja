use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use crate::error::{Error, ErrorKind};

/// Helper for tracking cumulative intermediate allocation during a render.
///
/// This mirrors the fuel tracker: it is always wrapped in an `Arc` so that it
/// can be shared across nested invocations of the template evaluation, and it
/// accounts a running total for the whole render.  Unlike fuel it is not
/// feature gated and is always compiled.
pub struct AllocTracker {
    // The maximum number of intermediate bytes allowed.  A cap of `0` means
    // unlimited.
    cap: usize,
    // The running total of charged bytes.
    used: AtomicUsize,
}

impl AllocTracker {
    /// Creates a new allocation tracker with the given cap in bytes.
    ///
    /// The tracker is always wrapped in an `Arc` so that it can be shared
    /// across nested invocations of the template evaluation.
    pub fn new(cap: usize) -> Arc<AllocTracker> {
        Arc::new(AllocTracker {
            cap,
            used: AtomicUsize::new(0),
        })
    }

    /// Charges `bytes` against the budget.
    ///
    /// A cap of `0` is treated as unlimited and never fails.  Otherwise, once
    /// the running total exceeds the cap an error is returned.
    pub fn charge(&self, bytes: usize) -> Result<(), Error> {
        if self.cap == 0 {
            return Ok(());
        }
        let total = self
            .used
            .fetch_add(bytes, Ordering::Relaxed)
            .saturating_add(bytes);
        if total > self.cap {
            return Err(Error::new(
                ErrorKind::InvalidOperation,
                "template allocation budget exceeded",
            ));
        }
        Ok(())
    }
}

#[cfg(all(test, feature = "serde", feature = "builtins"))]
mod tests {
    use crate::{Environment, ErrorKind};

    // Self-doubling accumulator.  Note that minijinja scopes a plain
    // `{% set %}` to a loop's body, so `{% set s = s + s %}` does not carry
    // across iterations; the namespace idiom is the real vector that keeps
    // doubling `s` and would otherwise allocate without bound.
    const SELF_DOUBLING: &str = "{% set ns = namespace(s='x') %}\
        {% for i in range(40) %}{% set ns.s = ns.s + ns.s %}{% endfor %}{{ 'ok' }}";

    #[test]
    fn test_alloc_budget_bounds_self_doubling() {
        // With a small budget the render must fail rather than allocate without
        // bound.  The budget trips well before `ns.s` grows large, so the test's
        // peak allocation stays under ~1MB and never risks OOM.
        let mut env = Environment::new();
        env.set_max_intermediate_size(Some(1_000_000));
        let err = env.render_str(SELF_DOUBLING, ()).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidOperation);
        assert!(
            err.to_string()
                .contains("template allocation budget exceeded"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_alloc_budget_bounds_many_distinct_strings() {
        // The cumulative budget also bounds many distinct large strings (here
        // via `*`), not just a single self-doubling accumulator.  `n` is a
        // runtime value so `'x' * n` is evaluated (and charged) on every
        // iteration instead of being constant-folded at compile time.
        let mut env = Environment::new();
        env.set_max_intermediate_size(Some(1_000_000));
        let ctx = std::collections::BTreeMap::from([("n", 100_000i64)]);
        let err = env
            .render_str(
                "{% set ns = namespace() %}\
                 {% for i in range(100) %}{% set ns.s = 'x' * n %}{% endfor %}{{ 'ok' }}",
                ctx,
            )
            .unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidOperation);
        assert!(
            err.to_string()
                .contains("template allocation budget exceeded"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_alloc_budget_allows_normal_render() {
        // A normal small template still renders fine with a budget configured,
        // proving the guard does not break ordinary string building.
        let mut env = Environment::new();
        env.set_max_intermediate_size(Some(1_000_000));
        let rv = env.render_str("{{ 'hello ' + 'world' }}", ()).unwrap();
        assert_eq!(rv, "hello world");
    }

    #[test]
    fn test_alloc_budget_disabled_by_default() {
        // Without a configured budget the accumulator is left to Part 1's
        // per-concat guard: it never OOMs, and small renders are unaffected.
        let env = Environment::new();
        assert_eq!(env.max_intermediate_size(), None);
        let rv = env.render_str("{{ 'a' + 'b' + 'c' }}", ()).unwrap();
        assert_eq!(rv, "abc");
    }
}
