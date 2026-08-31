use std::fmt;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use crate::error::{Error, ErrorKind};

/// Approximate heap cost of one value held in a container, in bytes.
///
/// A sequence of many tiny strings costs far more than the sum of its string
/// lengths: every element also carries a slot in the backing `Vec` plus its own
/// allocation header.  Charging string bytes alone under-counts such a
/// container by more than an order of magnitude, which is what let a retained
/// `split` result grow past the budget while charging almost nothing, so
/// per-element accounting adds this.
pub(crate) const CONTAINER_ELEMENT_COST: usize = 2 * std::mem::size_of::<crate::value::Value>();

/// Callback used by the size-checked helpers to test a prospective allocation.
///
/// Shared with helpers that are reachable from call paths without a `State`
/// (the printf/`str.format` machinery), which therefore cannot take one.
// Only the builtin filters consult a size check, so this is unused when they
// are compiled out.
#[allow(dead_code)]
pub(crate) type SizeCheck<'a> = &'a dyn Fn(usize) -> Result<(), Error>;

/// A [`fmt::Write`] sink that checks its accumulated length as it grows.
///
/// Used where an output's final size cannot be computed up front (`pprint`'s
/// `Debug` formatting), so the write aborts partway instead of materializing
/// the whole thing and failing afterwards.  `fmt::Write` can only report a
/// unit error, so the real error is stashed and recovered by [`finish`].
#[allow(dead_code)]
pub(crate) struct BudgetWriter<'a> {
    buf: String,
    check: SizeCheck<'a>,
    err: Option<Error>,
}

#[allow(dead_code)]
impl<'a> BudgetWriter<'a> {
    pub fn new(check: SizeCheck<'a>) -> BudgetWriter<'a> {
        BudgetWriter {
            buf: String::new(),
            check,
            err: None,
        }
    }

    pub fn with_capacity(cap: usize, check: SizeCheck<'a>) -> BudgetWriter<'a> {
        BudgetWriter {
            buf: String::with_capacity(cap),
            check,
            err: None,
        }
    }

    /// Returns the accumulated string, or the budget error that stopped it.
    pub fn finish(self, rv: fmt::Result) -> Result<String, Error> {
        match rv {
            Ok(()) => Ok(self.buf),
            Err(_) => Err(self.err.unwrap_or_else(|| {
                Error::new(ErrorKind::InvalidOperation, "failed to format value")
            })),
        }
    }

    /// Replaces `err` with the budget error, if one stopped a write.
    ///
    /// `fmt::Write` can only report a unit error, so a caller that routes its
    /// own `Result` through this writer recovers the real cause here.
    pub fn take_err(&mut self, err: Error) -> Error {
        self.err.take().unwrap_or(err)
    }

    /// Consumes the writer and returns what was written.
    pub fn into_string(self) -> String {
        self.buf
    }
}

impl fmt::Write for BudgetWriter<'_> {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        if let Err(err) = (self.check)(self.buf.len().saturating_add(s.len())) {
            self.err = Some(err);
            return Err(fmt::Error);
        }
        self.buf.push_str(s);
        Ok(())
    }
}

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
            return Err(budget_error());
        }
        Ok(())
    }

    /// Checks whether `bytes` would fit in the budget WITHOUT charging them.
    ///
    /// This is the pre-allocation guard.  [`charge`](Self::charge) can only run
    /// once an operation has already built its result, so it bounds the running
    /// total but not the peak of any single step: a filter that expands its
    /// input by a large constant factor allocates the whole result before the
    /// VM ever sees it.  Sites that can compute (or incrementally observe) the
    /// size of what they are about to allocate call this first, so an oversized
    /// single operation fails instead of allocating.
    ///
    /// It deliberately does not commit anything: the operation's result is
    /// still charged by the VM once it is produced, so a checked site is
    /// accounted exactly once rather than twice.
    pub fn check(&self, bytes: usize) -> Result<(), Error> {
        if self.cap == 0 {
            return Ok(());
        }
        let total = self.used.load(Ordering::Relaxed).saturating_add(bytes);
        if total > self.cap {
            return Err(budget_error());
        }
        Ok(())
    }
}

/// The error every budget guard fails with.
fn budget_error() -> Error {
    Error::new(
        ErrorKind::InvalidOperation,
        "template allocation budget exceeded",
    )
}

#[cfg(all(test, feature = "serde", feature = "builtins"))]
mod tests {
    use crate::{Environment, Error, ErrorKind};

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
    fn test_alloc_budget_bounds_tilde_concat() {
        // `~` is Jinja2's idiomatic string-concat operator and self-doubles
        // exactly like `+`, so the budget must bound it the same way.
        let mut env = Environment::new();
        env.set_max_intermediate_size(Some(1_000_000));
        let err = env
            .render_str(
                "{% set ns = namespace(s='x') %}\
                 {% for i in range(40) %}{% set ns.s = ns.s ~ ns.s %}{% endfor %}{{ 'ok' }}",
                (),
            )
            .unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidOperation);
        assert!(
            err.to_string()
                .contains("template allocation budget exceeded"),
            "unexpected error: {err}"
        );
    }

    // Renders `src` with a 1MB budget and asserts the render fails specifically
    // because the intermediate-allocation budget was exceeded (not an OOM).
    fn assert_budget_trips(src: &str) {
        let mut env = Environment::new();
        env.set_max_intermediate_size(Some(1_000_000));
        let err = env.render_str(src, ()).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidOperation);
        assert!(
            err.to_string()
                .contains("template allocation budget exceeded"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_alloc_budget_bounds_replace_filter() {
        // The `replace` filter grows super-linearly (`a` -> `aa` doubles the
        // count of `a`s), so filter results must be charged at the ApplyFilter
        // arm.  Bounded now that they are.
        assert_budget_trips(
            "{% set ns = namespace(s='a') %}\
             {% for i in range(40) %}{% set ns.s = ns.s | replace('a', 'aa') %}{% endfor %}{{ 'ok' }}",
        );
    }

    #[test]
    fn test_alloc_budget_bounds_format_filter() {
        // The `format` filter with repeated `%s` doubles per iteration; charged
        // at the ApplyFilter arm.
        assert_budget_trips(
            "{% set ns = namespace(s='a') %}\
             {% for i in range(40) %}{% set ns.s = '%s%s' | format(ns.s, ns.s) %}{% endfor %}{{ 'ok' }}",
        );
    }

    #[test]
    fn test_alloc_budget_bounds_block_set_capture() {
        // A block `{% set %}` captures emitted output into a string (EndCapture);
        // referencing the accumulator twice doubles it each iteration.  Charged
        // at the EndCapture arm.
        assert_budget_trips(
            "{% set ns = namespace(s='x') %}\
             {% for i in range(40) %}{% set ns.s %}{{ ns.s }}{{ ns.s }}{% endset %}{% endfor %}{{ 'ok' }}",
        );
    }

    #[test]
    fn test_alloc_budget_bounds_macro_call() {
        // A macro returns its rendered output as a string (pushed by the
        // CallFunction arm); a doubling macro applied to the accumulator grows
        // exponentially.  Charged now.
        assert_budget_trips(
            "{% macro dup(s) %}{{ s }}{{ s }}{% endmacro %}\
             {% set ns = namespace(s='x') %}\
             {% for i in range(40) %}{% set ns.s = dup(ns.s) %}{% endfor %}{{ 'ok' }}",
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

    // ── Single-operation bounds ───────────────────────────────
    //
    // The cases above all grow across MANY operations, which a running total
    // catches on its own.  The cases below each blow the budget in ONE
    // operation, which a running total cannot catch: it is only consulted once
    // the operation has already allocated its result.  They pass only because
    // the growth sites check the size they are about to allocate first.
    //
    // Every input below is deliberately sized so that if its guard regresses,
    // the test allocates gigabytes rather than failing -- which is exactly the
    // bug being guarded against, and is why these are worth their runtime.

    #[test]
    fn test_alloc_budget_bounds_single_replace() {
        // `replace` expands by `len(to) / len(from)`, a ratio the template
        // controls: 128 KiB of `a` against a 65,000-byte replacement asks for
        // ~7.9 GiB from ONE filter call costing one unit of fuel.  Charging the
        // result cannot help -- by then it exists.
        let src = format!(
            "{{% set boom = big | replace('a','{}') %}}ok",
            "z".repeat(65_000)
        );
        let mut env = Environment::new();
        env.set_max_intermediate_size(Some(1_000_000));
        let ctx = std::collections::BTreeMap::from([("big", "a".repeat(131_072))]);
        let err = env.render_str(&src, ctx).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidOperation);
        assert!(
            err.to_string()
                .contains("template allocation budget exceeded"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_alloc_budget_bounds_replace_with_in_budget_replacement() {
        // The replacement need not be a template literal.  Building it with a
        // filter keeps it under the budget (so it is allowed), and using it once
        // then asks for input_len x replacement_len.  Sized here so a regression
        // allocates ~13 GB.
        assert_budget_trips(
            "{% set pad = 'x' | indent(100000, true) %}\
             {% set boom = 'aaaaaaaaaa' * 13000 %}\
             {% set blown = boom | replace('a', pad) %}ok",
        );
    }

    #[test]
    fn test_alloc_budget_bounds_indent_width() {
        // `indent` materializes its width as a run of spaces, and the width is
        // an unbounded integer, so this asks for ~10 GB from a 40-byte template
        // with no input at all.
        assert_budget_trips("{% set boom = 'y' | indent(10000000000, true) %}ok");
    }

    #[test]
    fn test_alloc_budget_bounds_format_width() {
        // Same shape through printf padding: the width in a format spec is an
        // unbounded integer materialized as fill characters.
        assert_budget_trips("{% set boom = '%10000000000s' | format('y') %}ok");
    }

    #[test]
    fn test_alloc_budget_bounds_container_reservations() {
        // `slice` and `batch` reserve one slot per count, and the count is
        // caller-controlled and unrelated to the input's size.
        assert_budget_trips("{% set boom = [1] | slice(10000000000) %}ok");
        assert_budget_trips("{% set boom = [1] | batch(10000000000) %}ok");
    }

    #[test]
    fn test_alloc_budget_bounds_repeated_seq_join() {
        // `seq * n` is lazy, so nothing is allocated until `join` materializes
        // it -- in one operation, at n times the sequence's size.
        assert_budget_trips(
            "{% set row = 'x' * 100000 %}{% set boom = ([row] * 100000) | join('') %}ok",
        );
    }

    #[test]
    fn test_alloc_budget_bounds_map_over_growth_filter() {
        // `map` runs the per-element filter inside a single call, so the whole
        // sequence expands before the evaluator sees one result.
        assert_budget_trips(
            "{% set boom = ('a' * 20000) | split('a') | map('indent', 20000, true) %}ok",
        );
    }

    #[test]
    fn test_alloc_budget_bounds_pprint() {
        // `pprint` Debug-formats whatever it is handed, including a lazily
        // repeated sequence, with no size knowable in advance.
        assert_budget_trips(
            "{% set row = 'x' * 100000 %}{% set boom = ([row] * 100000) | pprint %}ok",
        );
    }

    #[test]
    fn test_alloc_budget_bounds_capture_block() {
        // A `{% set %}...{% endset %}` block accumulates into a buffer that
        // never reaches the caller's writer, so a bounded output sink cannot see
        // it, and charging the finished capture only reports a buffer already
        // built.  The buffer must be bounded while it grows.
        assert_budget_trips(
            "{% set big = 'x' * 100000 %}\
             {% set boom %}{% for i in range(100000) %}{{ big }}{% endfor %}{% endset %}ok",
        );
    }

    #[test]
    fn test_alloc_budget_bounds_macro_body() {
        // A macro body renders into its own buffer, with the same blind spot.
        assert_budget_trips(
            "{% macro blow(s) %}{% for i in range(100000) %}{{ s }}{% endfor %}{% endmacro %}\
             {% set big = 'x' * 100000 %}{% set boom = blow(big) %}ok",
        );
    }

    #[test]
    fn test_alloc_budget_charges_retained_containers() {
        // A filter that returns a container used to charge nothing at all,
        // because only the top-level value was inspected and a sequence is not
        // a string.  Splitting a string into single characters costs far more
        // than the string itself -- every element carries a slot and its own
        // allocation -- so the elements must be accounted, or these accumulate
        // without limit.
        assert_budget_trips(
            "{% set ns = namespace(acc=[]) %}{% set big = 'ab' * 20000 %}\
             {% for i in range(200) %}{% set ns.acc = ns.acc + [big | split('a')] %}{% endfor %}ok",
        );
    }

    #[test]
    fn test_alloc_budget_charges_host_callback_containers() {
        // The same hole via an unknown-method callback, which is how a host
        // supplies Python `str` methods: `split` hands back an owned
        // `Vec<String>` the engine did not build and cannot see into by shape.
        let mut env = Environment::new();
        env.set_max_intermediate_size(Some(1_000_000));
        env.set_unknown_method_callback(|_state, value, method, _args| {
            let Some(s) = value.as_str() else {
                return Err(Error::from(ErrorKind::UnknownMethod));
            };
            match method {
                "chars" => Ok(s.chars().map(|c| c.to_string()).collect()),
                _ => Err(Error::from(ErrorKind::UnknownMethod)),
            }
        });
        let ctx = std::collections::BTreeMap::from([("big", "x".repeat(20_000))]);
        let err = env
            .render_str(
                "{% set ns = namespace(acc=[]) %}\
                 {% for i in range(200) %}{% set ns.acc = ns.acc + [big.chars()] %}{% endfor %}ok",
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
    fn test_alloc_budget_charges_string_slices() {
        // Slicing builds a fresh string, and that arm charged nothing, so
        // retained slices grew without moving the budget at all.
        assert_budget_trips(
            "{% set ns = namespace(acc=[]) %}{% set big = 'x' * 100000 %}\
             {% for i in range(200) %}{% set ns.acc = ns.acc + [big[0:100000]] %}{% endfor %}ok",
        );
    }

    #[test]
    fn test_alloc_budget_allows_realistic_chat_template() {
        // The guards must not reject the templates this exists to run.  A
        // chat-template-shaped render using the growth filters normally stays
        // far inside the budget.
        let mut env = Environment::new();
        env.set_max_intermediate_size(Some(1_000_000));
        let ctx = std::collections::BTreeMap::from([("content", "Hello <there>, how are you?")]);
        let rv = env
            .render_str(
                "{% set parts = content | split(' ') %}\
                 {% set body = parts | join('_') | replace('_', ' ') %}\
                 {% set padded = '%12s' | format('role') %}\
                 {{ padded }}|{{ body | indent(2, true) }}|{{ parts | length }}",
                ctx,
            )
            .unwrap();
        assert_eq!(rv, "        role|  Hello <there>, how are you?|5");
    }
}
