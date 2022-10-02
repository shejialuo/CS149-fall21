#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <stdint.h>
#include <fstream>
#include <iostream>

#include "CycleTimer.h"

#if MKL_INSTALLED
#include "mkl.h"
#endif

#include "gemm_ispc.h"
#include "ref_gemm_ispc.h"

#define N_ITERS 3 // how many times to run implementaions for timing

// implement: C = alpha * A x B + beta * C
extern void gemm(int m, int n, int k, double *A, double *B, double *C, double alpha, double beta);

static float toBW(uint64_t bytes, float sec) {
    return static_cast<float>(bytes) / (1024. * 1024. * 1024.) / sec;
}

static float toGFLOPS(uint64_t ops, float sec) {
    return static_cast<float>(ops) / 1e9 / sec;
}

// Useful if you want to print a whole matrix
void printMat(const char * name, double *A, int m, int n){
    printf("--- %s ---\n", name);
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++) {
            printf("%.2lf ", A[i*n+j]);
        }
        printf("\n");
    }
    printf("\n");
    return;
}

// Allocate and populate matrices
int allocMatrices(int m, int n, int k, double **A, double **B, double **C){
#if MKL_INSTALLED
    // mkl_malloc aligns allocated memory on 64-byte boundaries for performance
    *A = (double *)mkl_malloc( m*k*sizeof( double ), 64 );
    *B = (double *)mkl_malloc( k*n*sizeof( double ), 64 );
    *C = (double *)mkl_malloc( m*n*sizeof( double ), 64 );
    if (*A == NULL || *B == NULL || *C == NULL) {
        // Could not allocate memory; abort
        return 1;
    }
#else
    *A = (double *)malloc( m*k*sizeof( double ));
    *B = (double *)malloc( k*n*sizeof( double ));
    *C = (double *)malloc( m*n*sizeof( double ));
    if (*A == NULL || *B == NULL || *C == NULL) {
        return 1;
    }
#endif
    return 0;
}

void fillMatrices(int m, int n, int k, double **A, double **B, double **C){
    // Populate the matrices with some data
    int i;
    for (i = 0; i < (m*k); i++) {
        (*A)[i] = ((double)rand() / (double)RAND_MAX);
    }

    for (i = 0; i < (k*n); i++) {
        (*B)[i] = ((double)rand() / (double)RAND_MAX);
    }

    for (i = 0; i < (m*n); i++) {
        (*C)[i] = ((double)rand() / (double)RAND_MAX);
    }
}

// Compute C=alpha*A*B+beta*C using Intel MKL and your implementation
int main(int argc, char *argv[]) {
    // Problem size calculations
    int m, n, k;
    int size = atoi(argv[1]);
    m = size, k = size, n = size;
    const uint64_t TOTAL_BYTES = (m*k + k*n + 2*m*n) * sizeof(double);
    const uint64_t TOTAL_FLOPS = 2*m*((uint64_t) n*k);

    double alpha, beta;
    alpha = 1.0; beta = 1.0;

    // Prepare matrices
    double *A1, *B1, *C1; // for MKL library implementation
    double *A2, *B2, *C2; // for your implementation
    double *A3, *B3, *C3; // for ispc implementation
    if(allocMatrices(m,n,k,&A1,&B1,&C1) == 1){
        return 1;
    }
    if(allocMatrices(m,n,k,&A2,&B2,&C2) == 1){
        return 1;
    }
    if(allocMatrices(m,n,k,&A3,&B3,&C3) == 1){
        return 1;
    }
    //
    // Repeat N_ITERS times for robust timing.
    //
#if MKL_INSTALLED
    double minMKL = 1e30;
    double totalsqerr_ispc = 0; // keep track of total squared error over all iterations
#endif
    double minGEMM = 1e30;
    double minISPC = 1e30;
    double totalsqerr_user = 0; // keep track of total squared error over all iterations

    printf("Running each implementation %d times...\n", N_ITERS);
    
    for (int i = 0; i < N_ITERS; ++i) {
        double startTime, endTime;

        // Fill input matrices with random data
        fillMatrices(m,n,k,&A1,&B1,&C1);
        
        // Make a copy of the matrices
        memcpy(A2,A1,m*k*sizeof(double));
        memcpy(B2,B1,k*n*sizeof(double));
        memcpy(C2,C1,m*n*sizeof(double));

        memcpy(A3,A1,m*k*sizeof(double));
        memcpy(B3,B1,k*n*sizeof(double));
        memcpy(C3,C1,m*n*sizeof(double));

        // Run the Intel MKL matrix multiply implementation.
#if MKL_INSTALLED
        printf("Running Intel MKL... ");
        startTime = CycleTimer::currentSeconds();
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                    m, n, k, alpha, A1, k, B1, n, beta, C1, n);
        endTime = CycleTimer::currentSeconds();
        printf("%.2lfms\n", (endTime - startTime)*1000);
        minMKL = std::min(minMKL, endTime - startTime);
#endif

        // Run your matrix multiply implementation. 
        printf("Running student GEMM... ");
        startTime = CycleTimer::currentSeconds();
        //ispc::gemm_ispc(m, n, k, A2, B2, C2, alpha, beta);
        gemm(m, n, k, A2, B2, C2, alpha, beta);
        endTime = CycleTimer::currentSeconds();
        printf("%.2lfms\n", (endTime - startTime)*1000);
        minGEMM = std::min(minGEMM, endTime - startTime);

        // Run reference ISPC matrix multiply implementation. 
        printf("Running ref ispc GEMM... ");
        startTime = CycleTimer::currentSeconds();
        ispc::gemm_ispc_ref(m, n, k, A3, B3, C3, alpha, beta);
        endTime = CycleTimer::currentSeconds();
        printf("%.2lfms\n", (endTime - startTime)*1000);
        minISPC = std::min(minISPC, endTime - startTime);

        // Compare output for correctness
        for (int i = 0; i < m; i++) {
            for ( int j = 0; j < n; j++ ) {
#if MKL_INSTALLED
                double mkl_output = C1[i*n+j];
                double sol_output = C2[i*n+j];
                double ispc_output = C3[i*n+j];
                totalsqerr_user += (mkl_output - sol_output) * (mkl_output - sol_output);
                totalsqerr_ispc += (mkl_output - ispc_output) * (mkl_output - ispc_output);
#else
                double sol_output = C2[i*n+j];
                double ispc_output = C3[i*n+j];
                totalsqerr_user += (ispc_output - sol_output) * (ispc_output - sol_output);
#endif
            }
        }
    }

    //
    // Report timing statistics
    //
#if MKL_INSTALLED
    printf("[Intel MKL]:\t\t[%.3f] ms\t[%.3f] GB/s\t[%.2f] GFLOPS\n",
           minMKL * 1000,
           toBW(TOTAL_BYTES, minMKL),
           toGFLOPS(TOTAL_FLOPS, minMKL));
#endif

    printf("[Student GEMM]:\t\t[%.3f] ms\t[%.3f] GB/s\t[%.2f] GFLOPS\n",
           minGEMM * 1000,
           toBW(TOTAL_BYTES, minGEMM),
           toGFLOPS(TOTAL_FLOPS, minGEMM));

    printf("[Ref ISPC GEMM]:\t[%.3f] ms\t[%.3f] GB/s\t[%.2f] GFLOPS\n",
           minISPC * 1000,
           toBW(TOTAL_BYTES, minISPC),
           toGFLOPS(TOTAL_FLOPS, minISPC));

    printf("Total squared error student sol: %lf\n", totalsqerr_user);
#if MKL_INSTALLED
    printf("Total squared error ref ispc: %lf\n", totalsqerr_ispc);
#endif

    // Deallocate matrices
#if MKL_INSTALLED
    mkl_free(A1);
    mkl_free(B1);
    mkl_free(C1);
    mkl_free(A2);
    mkl_free(B2);
    mkl_free(C2);
#else
    free(A1);
    free(B1);
    free(C1);
    free(A2);
    free(B2);
    free(C2);
#endif

    return 0;
}

