#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#define Max(a, b) ((a) > (b) ? (a) : (b))

#define N (2 * 2 * 2 * 2 * 2 * 2 * 2 + 2)
double maxeps = 0.1e-7;
int itmax = 100;
int i, j, k;
double w = 0.5;
double eps;

double A[N][N][N];

void relax();
void init();
void verify();

int main(int an, char **as)
{
    int it;
    printf("num_threads: %d, max: %d\n", omp_get_num_threads(), omp_get_max_threads());
#pragma omp parallel shared(A) // default is shared
    init();

    for (it = 1; it <= itmax; it++)
    {
        eps = 0.;
#pragma omp parallel private(i, j, k) shared(A, eps)
        relax();
        printf("it=%4i   eps=%f\n", it, eps);
        if (eps < maxeps)
            break;
    }

    verify();

    return 0;
}

void init() // для каждого процесса будут загружена страница памяти, содержащая используемые элементы
{
#pragma omp for private(i, j, k) schedule(static) collapse(3) 
    for (i = 0; i <= N - 1; i++)
        for (j = 0; j <= N - 1; j++)
            for (k = 0; k <= N - 1; k++)
            {
                if (i == 0 || i == N - 1 || j == 0 || j == N - 1 || k == 0 || k == N - 1)
                    A[i][j][k] = 0.;
                else
                    A[i][j][k] = (4. + i + j + k);
            }
}

void relax()
{
#pragma omp for schedule(static) collapse(2) reduction(max \
                                                       : eps)
    for (i = 1; i <= N - 2; i++)
        for (j = 1; j <= N - 2; j++)
            for (k = 1 + (i + j) % 2; k <= N - 2; k += 2)
            {
                double b;
                b = w * ((A[i - 1][j][k] + A[i + 1][j][k] + A[i][j - 1][k] + A[i][j + 1][k] + A[i][j][k - 1] + A[i][j][k + 1]) / 6. - A[i][j][k]);
                eps = Max(fabs(b), eps);
                A[i][j][k] = A[i][j][k] + b;
            }

#pragma omp barrier

#pragma omp for schedule(static) collapse(2)
    for (i = 1; i <= N - 2; i++)
        for (j = 1; j <= N - 2; j++)
            for (k = 1 + (i + j + 1) % 2; k <= N - 2; k += 2)
            {
                double b;
                b = w * ((A[i - 1][j][k] + A[i + 1][j][k] + A[i][j - 1][k] + A[i][j + 1][k] + A[i][j][k - 1] + A[i][j][k + 1]) / 6. - A[i][j][k]);
                A[i][j][k] = A[i][j][k] + b;
            }
}

void verify()
{
    double s;

    s = 0.;
#pragma omp parallel for private(i, j, k) shared(A) schedule(static) collapse(3) reduction(+ \
                                                                                           : s)
    for (i = 0; i <= N - 1; i++)
        for (j = 0; j <= N - 1; j++)
            for (k = 0; k <= N - 1; k++)
            {
                s = s + A[i][j][k] * (i + 1) * (j + 1) * (k + 1) / (N * N * N);
            }
    printf("  S = %f\n", s);
}
