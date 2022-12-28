### MPI

```
scp redb_3d_mpi.c skipod:. && ssh skipod
module load SpectrumMPI && mpicc redb_3d_mpi.c -o redb_3d_mpi -Wall -Werror &&  mpirun -np 2 redb_3d_mpi
```

### OpenMP

```
scp redb_3d_mpi.c skipod:. && ssh skipod
gcc ./redb_3d_openmp.c -o redb_3d_openmp -Ofast -fopenmp -Wall -Werror && time OMP_NUM_THREADS=16 ./redb_3d_openmp
```

### Python

```
python3 measure_speed.py redb_3d_openmp
```
