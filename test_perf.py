import numpy as np
import timeit

def bench_sum(resid, deta_a):
    return float((resid * deta_a).sum())

def bench_vdot(resid, deta_a):
    return float(np.vdot(resid, deta_a))

resid = np.random.randn(100, 100)
deta_a = np.random.randn(100, 100)

print("sum:", timeit.timeit(lambda: bench_sum(resid, deta_a), number=10000))
print("vdot:", timeit.timeit(lambda: bench_vdot(resid, deta_a), number=10000))
