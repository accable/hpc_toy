/* This code was taken from https://github.com/NVIDIA/NVPLSamples/blob/main/nvpl_blas/c/cgemm_batch_strided.c and 
 * and modified to accomodate scaled dot product Attention proposed by Vaswani et al., 2017. Unlike the cuTENSOR 
 * implementation at 2_0_multi_head_attention_cutensor.c, we do it right by using BLAS libraries to do the 
 * matrix multiplication for us.
 * 
 * Same with the previous code, scaled dot product Attention requires Q, K^t, and V values, we need to also declare 
 * the intermediate values too (i.e., S = Q * K^t, the attention kernel), and the output O.
 */

#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <math.h>
#include "nvpl_blas_cblas.h"

// The cluster now have perf but we still have this for flop calculation
#include <time.h>
#include <sys/resource.h>  		


// Memory profiling functions
long get_peak_memory_kb(){
	
	struct rusage usage;
	getrusage(RUSAGE_SELF, &usage);
	return usage.ru_maxrss;
}


// Applies softmax over K_len (last dimension of S[B, H, Q_len, K_len])
void softmax(float* S, int32_t B, int32_t H, int32_t Q, int32_t K){
    for(int b = 0; b < B; ++b){
        for(int h = 0; h < H; ++h){
            for(int q = 0; q < Q; ++q){
                
                // Get the pointer (we assume everything is 1D now)
                float* s_ptr = S + (((b * H + h) * Q + q) * K);

                // Get max
                float max_val = s_ptr[0];
                for (int k = 1; k < K; ++k){
                    if (s_ptr[k] > max_val) max_val = s_ptr[k];
                }

                // Get exponents and sum
                float sum = 0.0f;
                for (int k = 0; k < K; ++k){
                    s_ptr[k] = expf(s_ptr[k] - max_val);
                    sum += s_ptr[k];
                }

                // Normalize
                for (int k = 0; k < K; ++k){
                    s_ptr[k] /= sum;
                }
            }
        }
    }
}


/**
 * Taken from: https://github.com/NVIDIA/NVPLSamples/blob/main/nvpl_blas/c/example_helper.h
 * 
 * \brief      Fill data for single precision matrix
 *
 * \param      A        The pointer of the matrix
 * \param[in]  rows     The rows of the matrix
 * \param[in]  cols     The columns of the matrix
 * \param[in]  ld       The leading dimension of the matrix
 * \param[in]  order    row-major or column-major matrix
 * \param[in]  fillMode Matrix lower or upper or all the part is stored
 * \param[in]  diag     Indicates if the elements on the main diagonal of matrix are unity and should not be accessed
 */
static void fill_smatrix(float * A, nvpl_int_t rows, nvpl_int_t cols, nvpl_int_t ld, enum CBLAS_ORDER order,
        enum FILL_MODE fillMode, enum CBLAS_DIAG diag) {
    if (CblasColMajor == order) {
        if (Full == fillMode) {
            for (nvpl_int_t j = 0; j < cols; ++j) {
                for (nvpl_int_t i = 0; i < rows; ++i) {
                    A[i + ld * j] = (float)((float)1/1024.f * ((91u + j) & 1023u) - 0.5f);
                }
            }
        } else {
            for (nvpl_int_t j = 0; j < cols; ++j) {
                for (nvpl_int_t i = 0; i < rows; ++i) {
                    if (((Lower == fillMode) ? (j < i) : (j > i)) || ((CblasNonUnit == diag) && (i == j))) {
                        A[i + ld * j] = (float)((float)1/1024.f * ((91u + j) & 1023u) - 0.5f);
                    } else if (i == j) {
                        A[i + ld * j] = 1.0f;
                    } else {
                        A[i + ld * j] = NAN;
                    }
                }
            }
         }
    } else {
        if (Full == fillMode) {
            for (nvpl_int_t i = 0; i < rows; ++i) {
                for (nvpl_int_t j = 0; j < cols; ++j) {
                    A[j + ld * i] = (float)((float)1/1024.f * ((91u + i) & 1023u) - 0.5f);
                }
            }
        } else {
            for (nvpl_int_t i = 0; i < rows; ++i) {
                for (nvpl_int_t j = 0; j < cols; ++j) {
                    if (((Lower == fillMode) ? (j < i) : (j > i)) || ((CblasNonUnit == diag) && (i == j))) {
                        A[j + ld * i] = (float)((float)1/1024.f * ((91u + i) & 1023u) - 0.5f);
                    } else if (i == j) {
                        A[j + ld * i] = 1.0f;
                    } else {
                        A[j + ld * i] = NAN;
                    }
                }
            }
        }
    }
}


// START!
int main() 
{

    // Begin timer
	double time_start;
	time_start = clock();

    /**********************
     * Here we will be computing: 
     * S[B, H, Q_len, K_len] = alpha * Q[B, H, Q_len, D] K[B, H, D, K_len] + beta * S[B, H, Q_len, K_len]
     * 
     * and 
     * 
     * O[B, H, O_len, D] = alpha * softmax(S[B, H, Q_len, K_len]) V[B, H, V_len, D] + beta * O[B, H, O_len, D]
     * 
     * 
     * Tensor shapes assumed:
     * Q[B, H, Q_len, D]
     * K[B, H, D, K_len] (transposed because matmul)
     * S[B, H, Q_len, K_len] (stores Q and K^t matmul result)
     * V[B, H, V_len, D]
     * O[B, O_len, D] (output)
     * 
     * Tensor size: {1, 3, 1024, 64}
     **********************/

    nvpl_int_t q_len = 1024;  // Number of rows for Q and S (1024)
    nvpl_int_t k_len = 1024;  // Number of columns for K and S (1024)
    nvpl_int_t D = 64;  // Number of shared dimension (64)

    // Leading dimensions
    // So we can do tensor transformations (row/column major) without explicitly doing so (we let BLAS do it)
    nvpl_int_t ldq = 64;  // Because we wanted the data to be set as row-major
    nvpl_int_t ldk = 1024;   // Column major  
    nvpl_int_t lds = 64;   // Row-major, since the final tensor would be S[B, H, Q_len, K_len]

    nvpl_scomplex_t alpha = {1.0f, 1.0f};  // Set to 1
    nvpl_scomplex_t beta = {0.0f, 0.0f};  // And set to 0
    nvpl_int_t batch_size = 3;  // We set the batch size w.r.t heads so 3

    // Blas variables
    enum CBLAS_ORDER order = CblasRowMajor;  // Assumed to be row-major
    enum CBLAS_TRANSPOSE transA = CblasNoTrans;  
    enum CBLAS_TRANSPOSE transB = CblasNoTrans;  // No transpose because we already assumed K is transposed above

    // Memory pointers and size init
    float * Q = NULL;
    float * K = NULL;
    float * S = NULL;
    int64_t matrixSizeQ = 0;
    int64_t matrixSizeK = 0;
    int64_t matrixSizeS = 0;

    // Setting shape from tanspose conditions (if transpose x, else y)
    nvpl_int_t rowsQ = (transA == CblasNoTrans) ? q_len : k_len;
    nvpl_int_t colsQ = (transA == CblasNoTrans) ? k_len : q_len;
    nvpl_int_t rowsK = (transB == CblasNoTrans) ? k_len : D;
    nvpl_int_t colsK = (transB == CblasNoTrans) ? D : k_len;
    nvpl_int_t rowsS = q_len;
    nvpl_int_t colsS = k_len;

    // Get matrix size and the stride
    matrixSizeQ = (int64_t)ldq * colsQ;
    matrixSizeK = (int64_t)ldk * colsK;
    matrixSizeS = (int64_t)lds * colsS;
    nvpl_int_t strideQ = matrixSizeQ;
    nvpl_int_t strideK = matrixSizeK;
    nvpl_int_t strideS = matrixSizeS;

    printf("\nExample: cblas_cgemm_batch_strided for the matrix-matrix multiplication of a batch of matrices\n\n");
    printf("#### args: q_len=%" PRId64 ", k_len=%" PRId64 ", d=%" PRId64 ", ldq=%" PRId64 ", ldk=%" PRId64 ", lds=%" PRId64 ", "
            "transA=%c, transB=%c, order=%c\n", (int64_t)q_len, (int64_t)k_len, (int64_t)D, (int64_t)ldq, (int64_t)ldk,
            (int64_t)lds, transpose_to_char(transA), transpose_to_char(transB), order_to_char(order));
    printf("           strideQ=%" PRId64 ", strideK=%" PRId64 ", strideS=%" PRId64 ", batch_size=%" PRId64 "\n",
            (int64_t)strideQ, (int64_t)strideK, (int64_t)strideS, (int64_t)batch_size);
    printf("           alpha=(%g, %g), beta=(%g, %g)\n", alpha.real, alpha.imag, beta.real, beta.imag);

    // Allocating memory
    Q = (nvpl_scomplex_t *)malloc((strideQ * (int64_t)batch_size) * sizeof(nvpl_scomplex_t));
    K = (nvpl_scomplex_t *)malloc((strideK * (int64_t)batch_size) * sizeof(nvpl_scomplex_t));
    S = (nvpl_scomplex_t *)malloc((strideS * (int64_t)batch_size) * sizeof(nvpl_scomplex_t));

    // Assigning data
    for (nvpl_int_t i = 0; i < batch_size; ++i) {
        fill_smatrix(Q + strideQ * i, rowsQ, colsQ, ldq, order, Full, CblasNonUnit);
        fill_smatrix(K + strideK * i, rowsK, colsK, ldk, order, Full, CblasNonUnit);
        fill_smatrix(S + strideS * i, rowsS, colsS, lds, order, Full, CblasNonUnit);
    }

    // Call cgemm_batch_strided and end timer
    cblas_sgemm_batch_strided(order, transA, transB, q_len, k_len, D, &alpha, Q, ldq, strideQ,
            K, ldk, strideK, &beta, S, lds, strideS, batch_size);

    // End timer
	double elapsed_time;
	elapsed_time = (clock() - time_start)/CLOCKS_PER_SEC;
    long peak_mem = get_peak_memory_kb();  // Memory measurement

    // Calculating flops and outputting values
    int64_t flops_qk = 2LL * batch_size * q_len * D * k_len;;
    double gflops = (double)flops_qk / 1e9 / elapsed_time;

    printf("Simulation finished in: %f\n", elapsed_time);
    printf("Performance: %.2f GFLOPs\n", gflops);   
	printf("Peak memory usage: %ld KB (%.2f MB)\n", peak_mem, peak_mem / 1024.0);

    // Free memory
    free(Q);
    free(K);
    free(S);
    return 0;
}