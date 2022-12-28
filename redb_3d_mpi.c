#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#define Max(a, b) ((a) > (b) ? (a) : (b))

#define N (2 * 2 * 2 * 2 * 2 * 2 * 2 + 2)
#define REAL_INDEX(i_aux, startrow) (i_aux + startrow)
double maxeps = 0.1e-7;
int itmax = 100;
int i, j, k;
double w = 0.5;
double eps;

// double A[N][N][N];
typedef double type_array_2d[N][N];
type_array_2d *A; // we divide tensor A on n_procs parts and store each part in corresponding process

void relax();
void init();
void verify();

MPI_Request req[4];
int myrank, ranksize;
int startrow, lastrow, nrows;
MPI_Status status[4];

int ll, shift;

int main(int argc, char **argv)
{
	MPI_Init(&argc, &argv);					  /* initialize MPI system */
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);	  /* my place in MPI system */
	MPI_Comm_size(MPI_COMM_WORLD, &ranksize); /* size of MPI system */
	MPI_Barrier(MPI_COMM_WORLD);			  /* wait until all processes will be created */

	/* rows of matrix A to be processed */
	startrow = (myrank * (N - 2)) / ranksize; // we divide rows of matix A between processes. Indexation starts from real row of matrix A
	lastrow = (((myrank + 1) * (N - 2)) / ranksize) - 1;
	nrows = lastrow - startrow + 1;

	A = malloc((nrows + 2) * sizeof(type_array_2d)); // store matrix + upper and lower aux borders + left and right borders

	int it;
	init();

	for (it = 1; it <= itmax; it++) /* main loop */
	{
		eps = 0.;
		relax();
		if (myrank == 0)
			printf("it=%4i   eps=%f\n", it, eps);
		if (eps < maxeps)
			break;
	}

	verify();

	MPI_Finalize();
	return 0;
}

void init()
{
	for (int i_aux = 0; i_aux < nrows + 2; i_aux++)
		for (j = 0; j <= N - 1; j++)
			for (k = 0; k <= N - 1; k++)
			{
				i = REAL_INDEX(i_aux, startrow); // real index
				if (i == 0 || i == N - 1 || j == 0 || j == N - 1 || k == 0 || k == N - 1)
					A[i_aux][j][k] = 0.;
				else
					A[i_aux][j][k] = (4. + i + j + k);
			}
}

void relax()
{
	double local_eps = eps;
	for (i = 1; i <= nrows; i++) /* red cells */
		for (j = 1; j <= N - 2; j++)
			for (k = 1 + (REAL_INDEX(i, startrow) + j) % 2; k <= N - 2; k += 2)
			{
				double b;
				b = w * ((A[i - 1][j][k] + A[i + 1][j][k] + A[i][j - 1][k] + A[i][j + 1][k] + A[i][j][k - 1] + A[i][j][k + 1]) / 6. - A[i][j][k]);
				local_eps = Max(fabs(b), local_eps);
				A[i][j][k] = A[i][j][k] + b;
			}

	if (myrank != 0) /* exchanging borders */
		MPI_Irecv(&A[0], N * N, MPI_DOUBLE, myrank - 1, 1215, MPI_COMM_WORLD, &req[0]);
	if (myrank != ranksize - 1)
		MPI_Isend(&A[nrows], N * N, MPI_DOUBLE, myrank + 1, 1215, MPI_COMM_WORLD, &req[2]);
	if (myrank != ranksize - 1)
		MPI_Irecv(&A[nrows + 1], N * N, MPI_DOUBLE, myrank + 1, 1216, MPI_COMM_WORLD, &req[3]);
	if (myrank != 0)
		MPI_Isend(&A[1], N * N, MPI_DOUBLE, myrank - 1, 1216, MPI_COMM_WORLD, &req[1]);

	ll = 4, shift = 0;
	if (myrank == 0)
	{
		ll -= 2;
		shift = 2;
	}
	if (myrank == ranksize - 1)
	{
		ll -= 2;
	}

	MPI_Waitall(ll, &req[shift], &status[0]); /* waiting until all swaps will be done */

	for (int i = 1; i <= nrows; i++) /* black cells */
		for (j = 1; j <= N - 2; j++)
			for (k = 1 + (REAL_INDEX(i, startrow) + j + 1) % 2; k <= N - 2; k += 2)
			{
				double b;
				b = w * ((A[i - 1][j][k] + A[i + 1][j][k] + A[i][j - 1][k] + A[i][j + 1][k] + A[i][j][k - 1] + A[i][j][k + 1]) / 6. - A[i][j][k]);
				A[i][j][k] = A[i][j][k] + b;
			}

	if (myrank != 0) /* exchanging borders */
		MPI_Irecv(&A[0], N * N, MPI_DOUBLE, myrank - 1, 1215, MPI_COMM_WORLD, &req[0]);
	if (myrank != ranksize - 1)
		MPI_Isend(&A[nrows], N * N, MPI_DOUBLE, myrank + 1, 1215, MPI_COMM_WORLD, &req[2]);
	if (myrank != ranksize - 1)
		MPI_Irecv(&A[nrows + 1], N * N, MPI_DOUBLE, myrank + 1, 1216, MPI_COMM_WORLD, &req[3]);
	if (myrank != 0)
		MPI_Isend(&A[1], N * N, MPI_DOUBLE, myrank - 1, 1216, MPI_COMM_WORLD, &req[1]);

	ll = 4, shift = 0;
	if (myrank == 0)
	{
		ll -= 2;
		shift = 2;
	}
	if (myrank == ranksize - 1)
	{
		ll -= 2;
	}

	MPI_Waitall(ll, &req[shift], &status[0]); /* waiting until all swaps will be done */

	MPI_Allreduce(&local_eps, &eps, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD); /* updating global eps */
}

void verify()
{
	double s = 0, local_s = 0;

	for (i = 1; i <= nrows; i++)
		for (k = 0; k <= N - 1; k++)
			for (j = 0; j <= N - 1; j++)
			{
				local_s += A[i][j][k] * (REAL_INDEX(i, startrow) + 1) * (j + 1) * (k + 1) / (N * N * N);
			}

	MPI_Allreduce(&local_s, &s, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); /* updating global eps */
	if (myrank == 0)
		printf("  S = %f\n", s);
}