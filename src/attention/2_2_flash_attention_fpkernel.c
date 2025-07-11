/* Code referenced from https://github.com/tspeterkim/flash-attention-minimal/blob/main/flash.cu
 * Double checked w/ the paper: https://arxiv.org/abs/2205.14135
 * 
 * This is a "minimum viable implementation" of Flash-Attention forward-pass kernel in CUDA and CUTLASS.
 * This code assumes everything is done in fp16. 
*/

// Yes we are doing this.
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>  // Cutlass

#include <torch/types.h>  // Since we're piggybacking Torch's tensor


__global__ 
void flash_attn_forward(const __half* Q,  // Query
                        const __half* K,  // Key
                        const __half* V,  // Value
                        __half* O,  // Output 
                        const int B,  // Batch count
                        const int H,  // Head count
                        const int N,  // Q/K/V length
                        const int d,  // Dimension count
                        const int Tc,  // Thread column
                        const int Tr,  // Thread row
                        const int Bc,  // Block column
                        const int Br,  // Block row
                        const __half softmax_scale,  // Softmax w/ scaling (hence scaled Attention) 
                        __half* l, 
                        __half* m, 
                        const int WARP_SIZE  // GPU warp size
                        ){
    
    // Defining thread and blocks
    int tx = threadIdx.x;
    int bx = blockIdx.x;  // For batch
    int by = blockIdx.y;  // For head

    // Defining the offsets for QKV and lm
    int offset_qkv = (bx * gridDim.y * N * d) + (by * N * d);  // Reused by the output assuming they're the same
    int offset_lm = (bx * gridDim.y * N) + (by * N);  // Has no dimension

    // Defining SRAM and tiling for Q, K, V, and S
    // Unlike the original code, our Bc and Br is dynamic (follows the GPU size)
    extern __shared__ __half sram[];
    int Q_tile_size = Br * d;  // Tile size is Br * d each for Q
    int KV_tile_size = Bc * d;  // Tile size is Bc * d each for K and V

    // !!ALL OF THIS IS ON THE SHARED MEMORY!!
    __half* Q_tile = sram; 
    __half* K_tile = Q_tile + Q_tile_size;
    __half* V_tile = K_tile + KV_tile_size;
    __half* S_tile = V_tile + KV_tile_size;

    // Instead of initializing l and m as a single thing (i.e., m = -INFINITY), we initialize it 
    // w.r.t the length and offsetted by warp size
    for (int x = 0; x < N; x += WARP_SIZE){
        if (x + tx < N){
            l[offset_lm + tx + x] = 0;
            m[offset_lm + tx + x] = -INFINITY;
        }
    }                           


    // START
    // Outerloop where we load the K and V first
    for (int i = 0; i < Tc; i++){

        // Loading K and V to SRAM
        for (int x = 0; x < KV_tile_size; x += WARP_SIZE){
            if (x + tx < KV_tile_size){
                K_tile[tx + x] = K[offset_qkv + (KV_tile_size * i) + tx + x];
                V_tile[tx + x] = V[offset_qkv + (KV_tile_size * i) + tx + x];
            }
        }

        __syncthreads();  // So the inner loop can use the correct Kj and Vj

        // Inner loop where everything happens
        for (int j = 0; j < Tr; j++){

            // Loading Q to SRAM
            for (int x = 0; x < Q_tile_size; x += WARP_SIZE){
                Q_tile[tx + x] = Q[offset_qkv + (Q_tile_size * j) + tx + x];
            }

            __syncthreads();  // Again!

            // Getting previous m and l and current m and loading them to registers by
            // thread blocking
            __half row_m_prev;
            __half row_l_prev;

            if (tx < Br){
                row_m_prev = m[offset_lm + (Br * i) + tx];
                row_l_prev = l[offset_lm + (Br * i) + tx];
            }
            
            // STEP 1: S = Q * K^t
            // We moved to CUTLASS so it would be FAST (and in kernel too)
            // We set the M, N, K to 16, 16, and 16 respectively
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half, nvcuda::wmma::row_major> q_fragment;
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __half, nvcuda::wmma::col_major> k_fragment;  // Transposed
            nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, __half> s_fragment;
            nvcuda::wmma::fill_fragment(s_fragment, 0.0f);  // Allocating values to S

            for (int k = 0; k < d; k += 16){
                nvcuda::wmma::load_matrix_sync(q_fragment, Q_tile + k, d);
                nvcuda::wmma::load_matrix_sync(k_fragment, K_tile + k, d);
                nvcuda::wmma::mma_sync(s_fragment, q_fragment, k_fragment, s_fragment);
            }
            nvcuda::wmma::store_matrix_sync(S_tile, s_fragment, 16, nvcuda::wmma::mem_row_major);

            // STEP 2: Scaling and rowmax and P = exp(S - row_m) and row_l = rowsum(P) w/ thread blocking
            __half row_m = -INFINITY;
            __half row_l = 0;

            if (tx < Br){
                // Scaling and rowmax
                for (int i = 0; i < Bc; i++){
                    S_tile[(Bc * tx) + i] *= softmax_scale;
                    row_m = __hmax(row_m, S_tile[(Bc * tx) + i]);
                }

                // P = exp(S - row_m) and row_l = rowsum(P)
                for (int y = 0; y < Bc; y++){
                    S_tile[(Bc * tx) + y] = __expf(S_tile[(Bc * tx) + y] - row_m);
                    row_l += S_tile[(Bc * tx) + y];
                }
            }

            // STEP 3: Compute the rest (P * V) and new m amd l values and put back on HBM
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half, nvcuda::wmma::row_major> p_fragment;
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __half, nvcuda::wmma::row_major> v_fragment; 
            nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, __half> o_fragment;

            for (int x = 0; x < d; x += 16){
                nvcuda::wmma::fill_fragment(o_fragment, 0.0f);  // Filling initial values

                for (int k = 0; k < Br; k += 16){
                    nvcuda::wmma::load_matrix_sync(p_fragment, S_tile + k, Bc);
                    nvcuda::wmma::load_matrix_sync(v_fragment, V_tile + (k * d), Bc);
                    nvcuda::wmma::mma_sync(o_fragment, p_fragment, v_fragment, o_fragment);
                }
                nvcuda::wmma::store_matrix_sync(Q_tile + x, o_fragment, d, nvcuda::wmma::mem_row_major);
            }
            
            // Updating m and l values w/ thread blocking
            if (tx < Br){
                __half row_m_new = __hmax(row_m_prev, row_m);
                __half row_l_new = (hexp(row_m_prev - row_m_new) * row_l_prev) + (hexp(row_m - row_m_new) * row_l);

                for (int x = 0; x < d; x++){
                    O[offset_qkv + (KV_tile_size * j) + (tx * d) + x] = (__float2half(1.0f) / row_l_new) \
                    * ((row_l_prev * hexp(row_m_prev - row_m_new) *  O[offset_qkv + (KV_tile_size * j) + (tx * d) + x]) \
                    + (hexp(row_m - row_m_new) * Q_tile[(tx * d) + x]));
                }
                m[offset_lm + (Br * j) + tx] = row_m_new;
                l[offset_lm + (Br * j) + tx] = row_l_new;
            }
        }
        __syncthreads();
    }
}


// Defining the forward pass w/ libtorch
torch::Tensor forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V){
    // ========== Host side code ==========
    // Get tensor properties (batch, head, len, dim)
    const int B = Q.size(0);
    const int H = Q.size(1);
    const int N = Q.size(2);
    const int d = Q.size(3);

    // To get Br and Bc dynamically, we need to get the GPU properties
    // Doing so allows the code to automatically the *biggest permitted block size*
    // We also added warp size despite always being 32
    // For reference, this was run on 1x V100 16GB
    int gpu_sram_size;
    int gpu_warp_size;
    cudaDeviceGetAttribute(&gpu_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    cudaDeviceGetAttribute(&gpu_warp_size, cudaDevAttrWarpSize, 0);
    
    int max_tile = ceil(gpu_sram_size / (sizeof(__half) * 4 * d));
    const int Bc = max_tile;
    const int Br = std::min(max_tile, d);

    // Check if the Bc and Br is within GPU limits
    const int used_sram_size = (3 * Bc * d + Bc * Br * sizeof(__half));
    printf("Br size: %d \n", Br);
    printf("Bc size: %d \n", Bc);
    printf("GPU shared memory size: %d \n", gpu_sram_size);
    printf("Requested shared memory size: %d \n", used_sram_size);

    // Get Tc and Tr
    const int Tc = ceil((float) N / Bc);
    const int Tr = ceil((float) N / Br);

    // Softmax scale
    const float softmax_scale = 1.0 / sqrt(d);  // Scaled Dot-Product Attention

    // ========== Device side code ==========
    // Initialize O, l, and m to HBM
    // Since we're using Libtorch for Q, K, and V, m and l uses cudamemalloc
    auto O = torch::zeros_like(Q);
    __half* l = NULL;
    __half* m = NULL;
    // Both has no dimensions!
    cudaMalloc((void**) &l, B * H * N * sizeof(__half));
    cudaMalloc((void**) &m, B * H * N * sizeof(__half));
    
    dim3 grid_dim(B, H);
    dim3 block_dim(gpu_warp_size);

    flash_attn_forward<<<grid_dim, block_dim, used_sram_size>>>(
        reinterpret_cast<__half*>(Q.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(K.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(V.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(O.data_ptr<at::Half>()),
        B, 
        H, 
        N, 
        d, 
        Tc,
        Tr,
        Bc,
        Br,
        __float2half(softmax_scale), 
        l, 
        m, 
        gpu_warp_size
    );

    return O;
}