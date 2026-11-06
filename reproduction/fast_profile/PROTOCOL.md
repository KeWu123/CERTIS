# Exact piecewise profile validation

This exploratory extension follows the supplied external review. The review's
claimed validation counts and acceptance probabilities are not experiment data.
The released v2 certificate, allocator and historical records remain unchanged.

Before incorporating a method claim, compare every profile entry AND attaining
count pair against exhaustive Python construction and guarded NumPy construction.
Include forced old discards, zero unknowns, census/empty audits, clipped bounds,
zero/one allowance, large exact integer denominators and ties. Separately test
quadratic-envelope candidate minimization against full integer enumeration.

Measure cold and query-local memoized old/new construction with identical states
and matching tie-breaking. Include repeated uniform blocks and heterogeneous,
imbalanced, skewed and repeated-refinement states. Record negative speed results.
The new builder additionally rechecks each selected witness with the four-pair
formula; the old builder evaluates that kernel throughout enumeration. Full
profile equality checks are outside timing. No change in oracle requests or
quality follows from replacing an identical profile.

Do not substitute an arithmetic-operation bound for bit complexity or for the
global DP convolution cost. Do not label a source-equivalent prototype as an
already established empirical contribution.

After observing the pure Python prototype's overhead, add a NumPy implementation
with the same candidate theorem and exact isqrt neighbours. A Python-integer
guard rejects unsafe int64 coefficient/discriminant ranges. Predeclare a fixed
small-left cutoff of 1,024 for short vectorized enumeration loops; all larger
safe states use piecewise search, and unsafe states use Python piecewise search.
Run the same 14 cases into a distinct output directory, retaining all initial
pure Python timings. This is an exploratory implementation revision, not a
pre-registered speed hypothesis or a change in the quality evaluation.
