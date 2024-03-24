#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>

#define N 5 // Size of the matrix

int main() {
    // Define and initialize the original matrix
    double data[] = {4.0, 1.0, 1.0, 0.0, 1.0, -1.0, -3.0, 1.0, 1.0, 0.0, 2.0, 1.0, 5.0, -1.0, -1.0, -1.0, 4.0, 0.0, 0.0, 2.0, -1.0, 1.0, 4.0};
    gsl_matrix_view A = gsl_matrix_view_array(data, N, N);
    
    // Print the original matrix A
    printf("Original Matrix A:\n");
    gsl_matrix_fprintf(stdout, &A.matrix, "%g");
    printf("\n");

    // Perform LU decomposition
    gsl_permutation *p = gsl_permutation_alloc(N);
    int signum;
    gsl_linalg_LU_decomp(&A.matrix, p, &signum);

    // Allocate memory for matrices L, U, and LU
    gsl_matrix *L = gsl_matrix_alloc(N, N);
    gsl_matrix *U = gsl_matrix_alloc(N, N);
    gsl_matrix *LU = gsl_matrix_alloc(N, N);

    // Copy lower and upper triangular parts from the LU decomposition
    gsl_matrix_set_identity(L);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (i > j) {
                gsl_matrix_set(L, i, j, gsl_matrix_get(&A.matrix, i, j));
            } else {
                gsl_matrix_set(U, i, j, gsl_matrix_get(&A.matrix, i, j));
            }
        }
    }

    // Reconstruct the matrix A from L and U
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, L, U, 0.0, LU);


    // Print matrices L, U, and LU
    printf("Matrix L:\n");
    gsl_matrix_fprintf(stdout, L, "%g");
    printf("\n");

    printf("Matrix U:\n");
    gsl_matrix_fprintf(stdout, U, "%g");
    printf("\n");

    printf("Reconstructed Matrix LU:\n");
    gsl_matrix_fprintf(stdout, LU, "%g");
    printf("\n");

    // Free allocated memory
    gsl_permutation_free(p);
    gsl_matrix_free(L);
    gsl_matrix_free(U);
    gsl_matrix_free(LU);

    return 0;
}

/*
Solution
Original Matrix A:
3
-1
1
3
6
2
3
3
7

Matrix L:
1
0
0
1
1
0
1
0.571429
1

Matrix U:
3
-1
1
0
7
1
0
0
5.42857

Reconstructed Matrix LU:
3
-1
1
3
6
2
3
3
7
Solution2:
Original Matrix A:
10
-1
0
-1
10
-2
0
-2
10

Matrix L:
1
0
0
-0.1
1
0
0
-0.20202
1

Matrix U:
10
-1
0
0
9.9
-2
0
0
9.59596

Reconstructed Matrix LU:
10
-1
0
-1
10
-2
0
-2
10

Solution3:
Original Matrix A:
10
5
0
0
5
10
-4
0
-4
8
1
0
0
-1
5
3.80298e-106

Matrix L:
1
0
0
0
-0.4
1
0
0
0
-0.1
1
0
0.5
0.75
-0.931373
1

Matrix U:
10
5
0
0
0
10
1
0
0
0
5.1
3.80298e-106
0
0
0
3.54199e-106

Reconstructed Matrix LU:
10
5
0
0
-4
8
1
0
0
-1
5
3.80298e-106
5
10
-4
0

*** stack smashing detected ***: terminated
Aborted (core dumped)

Solution 4:
4
1
1
0
1
-1
-3
1
1
0
2
1
5
-1
-1
-1
4
0
0
2
-1
1
4
7.63994e-302
0

Matrix L:
1
0
0
0
0
-0.25
1
0
0
0
0.5
0.117647
1
0
0
-0.25
-0.647059
0.315789
1
0
-0.25
0.294118
0.934211
0.71
1

Matrix U:
4
1
1
0
1
0
4.25
0.25
0
2.25
0
0
4.47059
-1
-1.76471
0
0
0
1.31579
2.26316
0
0
0
0
-0.37

Reconstructed Matrix LU:
4
1
1
0
1
-1
4
0
0
2
2
1
5
-1
-1
-1
-3
1
1
0
-1
1
4
0
0

*** stack smashing detected ***: terminated
Aborted (core dumped)

*/
