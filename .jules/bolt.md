## 2024-03-25 - Pre-compiling Regex in performance-critical loops
**Learning:** Initializing `re` matches inside loops without pre-compiling adds significant overhead. Profiling regex performance specifically in `parse_charge_mult` showed that dynamic matching creates a ~1.5x-2x performance bottleneck over 100k invocations compared to `re.compile()` at the module level.
**Action:** Always extract regex expressions into pre-compiled module-level constants (e.g., `RE_CHARGE`, `RE_XYZ`) instead of defining them inline, especially in frequently called parsing loops.
## 2025-04-02 - Vectorized operations
**Learning:** Replaced nested pure Python loops `sum(...) / ...` with vectorized operations like `np.mean(coords, axis=0).tolist()` and `np.average(coords, weights=Z, axis=0).tolist()` for center of geometry and charge calculation.
**Action:** Always prefer `numpy` over pure Python for simple nested reductions when arrays of floats are involved. Remember that `.tolist()` converts it seamlessly back to the expected output type.
