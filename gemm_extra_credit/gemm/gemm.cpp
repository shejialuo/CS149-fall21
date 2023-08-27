#include <cstdio>

/**
 * @brief Gemm -- general double precision dense matrix-matrix multiplication
 *
 * @details Implement: C = alpha * A x B + beta * C, for matrices A, B, C.
 * Matrix C is M x N  (M rows, N columns)
 * Matrix A is M x K
 * Matrix B is K x N
 *
 * @author shejialuo
 */

/**
 * @brief `Gemm` class is the most simplest implementation, it uses the native
 * way to calculate the matrix-matrix multiplication.
 *
 */
class Gemm {
public:
  /**
   * @brief Disable constructor
   *
   */
  Gemm() = delete;

  /**
   * @brief Calculate the GEMM sequentially
   *
   * @details The matrix A, B, C are all one-dimensional array, so the need
   * to find the way to calculate the matrix multiplication.
   *
   */
  static void gemm(int m, int n, int k, double *A, double *B, double *C,
                   double alpha, double beta) {
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        double inner_pod = 0;
        for (int kk = 0; kk < k; kk++) {
          inner_pod += A[i * k + kk] * B[kk * n + j];
        }
        C[i * n + j] = alpha * inner_pod + beta * C[i * n + j];
      }
    }
  }
};

/**
 * @brief A wrapper to wrap the coordinates for indicating the
 * start left-top point for the current blocked matrix.
 *
 */
struct Point2D {
  int i{}; /**< The absolute i */
  int j{}; /**< The absolute j */
};

/**
 * @brief `GemmBlock` class is a class to wrap the functions to implement
 * the dense matrix-matrix multiplication with blocked idea.
 *
 */
class GemmBlock {
public:
  /**
   * @brief Disable constructor
   *
   */
  GemmBlock() = delete;

  /**
   * @brief Apply the C = beta C, it should be calculated at first.
   *
   */
  static void addMatrix(int mBlock, int nBlock, int n, Point2D &c, double *C,
                        double beta) {
    for (int i = 0; i < mBlock; i++) {
      for (int j = 0; j < nBlock; j++) {
        C[(i + c.i) * n + (j + c.j)] = beta * C[(i + c.i) * n + (j + c.j)];
      }
    }
  }

  /**
   * @brief Matrix multiplication with block
   *
   * @details In order to implement the block matrix multiplication, we need
   * to calculate the blocked matrix A and blocked matrix B. For example
   * A11 A12  B11 B12  C11 C12
   * A21 A22  B21 B22  C21 C22
   * C11 = A11 X b11 + A12 X B21
   *
   */
  static void matrixMultiplicationBlock(int mBlock, int nBlock, int kBlock,
                                        int n, int k, Point2D &a, Point2D &b,
                                        Point2D &c, double *A, double *B,
                                        double *C, double alpha, double beta) {

    for (int i = 0; i < mBlock; i++) {
      for (int j = 0; j < nBlock; j++) {
        double inner_pod = 0;
        for (int kk = 0; kk < kBlock; kk++) {
          inner_pod +=
              A[(i + a.i) * k + (kk + a.j)] * B[(kk + b.i) * n + (j + b.j)];
        }
        C[(i + c.i) * n + (j + c.j)] += alpha * inner_pod;
      }
    }
  }

  /**
   * @brief Split the matrix
   *
   */
  static void gemmUsingBlock(int m, int n, int k, double *A, double *B,
                             double *C, double alpha, double beta) {
    Point2D a{}, b{}, c{};
    const int size = 6;
    for (int i = 0; i < m; i += size) {
      int mBlock = i + size < m ? size : m - i;
      a.i = i;
      c.i = i;
      for (int j = 0; j < n; j += size) {
        b.j = j;
        c.j = j;
        int nBlock = j + size < n ? size : n - j;
        addMatrix(mBlock, nBlock, n, c, C, beta);
        for (int kk = 0; kk < k; kk += size) {
          a.j = kk;
          b.i = kk;
          int kBlock = kk + size < k ? size : k - kk;
          matrixMultiplicationBlock(mBlock, nBlock, kBlock, n, k, a, b, c, A, B,
                                    C, alpha, beta);
        }
      }
    }
  }

  static void gemm(int m, int n, int k, double *A, double *B, double *C,
                   double alpha, double beta) {
    gemmUsingBlock(m, n, k, A, B, C, alpha, beta);
  }
};

class GemmBlockWithMemoryLayoutChange {
public:
  /**
   * @brief Disable constructor
   *
   */
  GemmBlockWithMemoryLayoutChange() = delete;

  static void addMatrix(int mBlock, int nBlock, int n, Point2D &c, double *C,
                        double beta) {
    for (int i = 0; i < mBlock; i++) {
      for (int j = 0; j < nBlock; j++) {
        C[(i + c.i) * n + (j + c.j)] = beta * C[(i + c.i) * n + (j + c.j)];
      }
    }
  }

  static void matrixMultiplicationBlock(int mBlock, int nBlock, int kBlock,
                                        int n, int k, Point2D &a, Point2D &b,
                                        Point2D &c, double *A, double *B,
                                        double *C, double alpha, double beta) {

    for (int i = 0; i < mBlock; i++) {
      for (int j = 0; j < nBlock; j++) {
        double inner_pod = 0;
        for (int kk = 0; kk < kBlock; kk++) {
          inner_pod +=
              A[(i + a.i) * k + (kk + a.j)] * B[(j + b.i) * n + (kk + b.j)];
        }
        C[(i + c.i) * n + (j + c.j)] += alpha * inner_pod;
      }
    }
  }

  static void gemmUsingBlock(int m, int n, int k, double *A, double *B,
                             double *C, double alpha, double beta) {
    Point2D a{}, b{}, c{};
    const int size = 6;
    for (int i = 0; i < m; i += size) {
      int mBlock = i + size < m ? size : m - i;
      a.i = i;
      c.i = i;
      for (int j = 0; j < n; j += size) {
        b.i = j;
        c.j = j;
        int nBlock = j + size < n ? size : n - j;
        addMatrix(mBlock, nBlock, n, c, C, beta);
        for (int kk = 0; kk < k; kk += size) {
          a.j = kk;
          b.j = kk;
          int kBlock = kk + size < k ? size : k - kk;
          matrixMultiplicationBlock(mBlock, nBlock, kBlock, n, k, a, b, c, A, B,
                                    C, alpha, beta);
        }
      }
    }
  }

  static void gemm(int m, int n, int k, double *A, double *B, double *C,
                   double alpha, double beta) {

    double *newB = new double[n * k];

    for (int j = 0; j < n; j++) {
      for (int i = 0; i < k; i++) {
        newB[j * k + i] = B[i * n + j];
      }
    }

    gemmUsingBlock(m, n, k, A, newB, C, alpha, beta);

    delete[] newB;
  }
};

void gemm(int m, int n, int k, double *A, double *B, double *C, double alpha,
          double beta) {
  // Brute Force
  // Gemm::gemm(m, n, k, A, B, C, alpha, beta);

  // SubMatrix Multiplication
  // GemmBlock::gemm(m, n, k, A, B, C, alpha, beta);

  // SubMatrix Multiplication with B memory layout change
  GemmBlockWithMemoryLayoutChange::gemm(m, n, k, A, B, C, alpha, beta);
}
