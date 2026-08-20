## Fixed

- Model resolution now admits only exact package-owned exploratory and confirmatory model records before reading their fields, so caller-defined model-spec subclasses cannot execute attribute callbacks during validation. Exact built-in/concrete NumPy factor counts and exact package model records retain their existing behavior; multidimensional exploratory estimation remains separately governed by #633.
