
// gemm -- general double precision dense matrix-matrix multiplication.
//
// implement: C = alpha * A x B + beta * C, for matrices A, B, C
// Matrix C is M x N  (M rows, N columns)
// Matrix A is M x K
// Matrix B is K x N
//
// Your implementation should make no assumptions about the values contained in any input parameters.

void gemm(int m, int n, int k, double *A, double *B, double *C, double alpha, double beta){

    // REPLACE THIS WITH YOUR IMPLEMENTATION

    int i, j, kk;
    for (i=0; i<m; i++){
        for (j=0; j<n; j++){
	    double inner_prod = 0;
	    for (kk=0; kk<k; kk++){
	        inner_prod += A[i*k+kk] * B[kk*n+j];
	    }
	    C[i*n+j] = alpha * inner_prod + beta * C[i*n+j];
	}
    }
    
    // END OF NAIVE IMPLEMENTATION
}

