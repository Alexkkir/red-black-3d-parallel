import time
import os, sys

file = sys.argv[1]
framework = sys.argv[2]
assert framework in ['openmp', 'mpi']
threads_space = [1, 2, 4, 8, 16, 32, 64, 128]
results = []
out_vals = []
tmp_file = 'tmp.txt'

n_runs = 3



for n_threads in threads_space:
    out_vals.append([])

    start = time.time()
    for _ in range(n_runs):
        if framework == 'openmp': 
            os.system(f"OMP_NUM_THREADS={n_threads} ./{file} > {tmp_file}")
        else:
            os.system(f"mpirun -np {n_threads} ./{file} > {tmp_file}")
        with open(tmp_file) as f:
            out = f.readlines()[-1].split()[-1]
        out_vals[-1].append(out)
    end = time.time()

    t = (end - start) / n_runs
    results.append(t)
    print(n_threads, t)

print(threads_space)
print(results)
print(out_vals)