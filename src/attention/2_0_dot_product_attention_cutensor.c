/* This code was taken from https://github.com/NVIDIA/NVPLSamples/blob/main/nvpl_tensor/contraction/contraction.c and 
* and modified to accomodate scaled dot product Attention proposed by Vaswani et al., 2017. This however, is not 
* considered as a "correct" implementation of Attention, as this was more of an experimentation rather than "actually 
* coding Attention". Since cuTENSOR is not made for our purposes, we only present the "largest configuration you can run 
* without cuTENSOR whining and throwing fit" configuration. 
* 
* Segmentation fault problem when assigning large sized extents from the original code has been fixed in this code.
* 
* Since scaled dot product Attention requires Q, K^t, and V values, we need to also declare the intermediate values too
* (i.e., S = Q * K^t, the attention kernel), and the output O.
*/

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <nvpl_tensor.h>

// Since the cluster does not have perf or gperf, we need to do this the old fashioned way
#include <time.h>
#include <sys/resource.h>  		

// Memory profiling functions
long get_peak_memory_kb(){
	
	struct rusage usage;
	getrusage(RUSAGE_SELF, &usage);
	return usage.ru_maxrss;
}


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


// START!
#define HANDLE_ERROR(x)                                           \
    {                                                             \
        const nvpltensorStatus_t err = x;                         \
        if (err != NVPLTENSOR_STATUS_SUCCESS)                     \
        {                                                         \
            printf("Error: %s\n", nvpltensorGetErrorString(err)); \
            exit(-1);                                             \
        }                                                         \
    };


int main()
{

    // Begin timer
	double time_start;
	time_start = clock();

    typedef float floatTypeQ;  // Query
    typedef float floatTypeK;  // Key
    typedef float floatTypeS;  // Attention Kernel
    typedef float floatTypeV;  // value
    typedef float floatTypeO;  // Output
    typedef float floatTypeCompute;

    nvpltensorDataType_t typeQ = NVPLTENSOR_R_32F;
    nvpltensorDataType_t typeK = NVPLTENSOR_R_32F;
    nvpltensorDataType_t typeS = NVPLTENSOR_R_32F;
    nvpltensorDataType_t typeV = NVPLTENSOR_R_32F;
    nvpltensorDataType_t typeO = NVPLTENSOR_R_32F;
    nvpltensorComputeDescriptor_t const descCompute = NVPLTENSOR_COMPUTE_DESC_32F;  // We keep it at 32 float just in case

    uint32_t const numThreads = 4;  // Set it to 4 first

    /**********************
     * Computing: S[B, H, Q_len, K_len] = alpha * Q[B, H, Q_len, D] K[B, H, D, K_len] + beta * S[B, H, Q_len, K_len]
     * 
     * Since creating the kernel requires one to do contraction first (literally matmul the thing), we can keep this code 
     * but modify the shapes instead!
     * 
     * Tensor shapes assumed:
     * Q[B, H, Q_len, D]
     * K[B, H, D, K_len] (transposed because matmul)
     * S[B, H, Q_len, K_len] (stores Q and K^t matmul result)
     * V[B, H, V_len, D]
     * O[B, O_len, D] (output)
     **********************/

    floatTypeCompute alpha = (floatTypeCompute) 1.0f;  // We set it to 1.0f
    floatTypeCompute beta = (floatTypeCompute) 0.0f;  // Is 0.0f so C won't be used

    int32_t modeQ[] = {0, 1, 2, 3};  // Q[B, H, Q_len, D]
    int32_t modeK[] = {0, 1, 3, 4};  // K[B, H, D, K_len]
    int32_t modeS[] = {0, 1, 2, 4};  // S[B, H, Q_len, K_len]
    int32_t modeV[] = {0, 1, 2, 3};  // V[B, H, O_len, D]  (the same as the query because self-attention)
    int32_t modeO[] = {0, 1, 2, 3};  // O[B, H, len, D]  (should follow the original input size with later flattened (H * D))

    enum { nmodeQ = 4 };
    enum { nmodeK = 4 };
    enum { nmodeS = 4 };
    enum { nmodeV = 4 };
    enum { nmodeO = 4 };

    /* Since this is using cuTENSOR and with how things were designed, this is the maximum size we can use before
    * cuTENSOR complaining that it cannot do the operation.
    *
    * Current assumption is D is 64 and since at the end it would be flattened (H * D), we assume dimension is 64,
    * head is 8, and the "on tensor" dimension being 8.
    */

    int64_t extent[] = {1, 8, 512, 8, 512};  // [B, H, len, D, len]

    int64_t extentQ[nmodeQ];
    for (int i = 0; i < nmodeQ; ++i)
    {
        extentQ[i] = extent[modeQ[i]];
    }
    int64_t extentK[nmodeK];
    for (int i = 0; i < nmodeK; ++i)
    {
        extentK[i] = extent[modeK[i]];
    }
    int64_t extentS[nmodeS];
    for (int i = 0; i < nmodeS; ++i)
    {
        extentS[i] = extent[modeS[i]];
    }
    int64_t extentV[nmodeV];
    for (int i = 0; i < nmodeV; ++i)
    {
        extentV[i] = extent[modeV[i]];
    }
    int64_t extentO[nmodeO];
    for (int i = 0; i < nmodeO; ++i)
    {
        extentO[i] = extent[modeO[i]];
    }

    /**********************
     * Allocating data
     * 
     * We also fixed a bug (more like oversight) from the sample code that would cause the calculation to overflow at 
     * scale (large values)
     **********************/

    int64_t elementsQ = 1;
    for (int i = 0; i < nmodeQ; ++i)
    {
        elementsQ *= extentQ[i];
    }
    int64_t elementsK = 1;
    for (int i = 0; i < nmodeK; ++i)
    {
        elementsK *= extentK[i];
    }
    int64_t elementsS = 1;
    for (int i = 0; i < nmodeS; ++i)
    {
        elementsS *= extentS[i];
    }
    int64_t elementsV = 1;
    for (int i = 0; i < nmodeV; ++i)
    {
        elementsV *= extentV[i];
    }
    int64_t elementsO = 1;
    for (int i = 0; i < nmodeO; ++i)
    {
        elementsO *= extentO[i];
    }

    int64_t sizeQ = sizeof(floatTypeQ) * elementsQ;
    int64_t sizeK = sizeof(floatTypeK) * elementsK;
    int64_t sizeS = sizeof(floatTypeS) * elementsS;
    int64_t sizeV = sizeof(floatTypeV) * elementsV;
    int64_t sizeO = sizeof(floatTypeO) * elementsO;

    uint32_t const kAlignment = 128;  // Alignment of the pointers (bytes)

    floatTypeQ* Q = aligned_alloc(kAlignment, sizeQ);
    floatTypeK* K = aligned_alloc(kAlignment, sizeK);
    floatTypeS* S = aligned_alloc(kAlignment, sizeS);
    floatTypeV* V = aligned_alloc(kAlignment, sizeV);
    floatTypeO* O = aligned_alloc(kAlignment, sizeO);

    if (Q == NULL || K == NULL || V == NULL)  // We only care about Q, K, and V since S and O can be empty for all we care
    {
        printf("Error: allocation of tensor memory.\n");
        return -1;
    }

    /*******************
     * Initialize data
     * 
     * Obviously we only initialize Q, K, and V and not S and O
     *******************/

    for (int64_t i = 0; i < elementsQ; i++)
        Q[i] = (((floatTypeQ) (rand() / RAND_MAX)) - 0.5) * 100;
    for (int64_t i = 0; i < elementsK; i++)
        K[i] = (((floatTypeK) (rand() / RAND_MAX)) - 0.5) * 100;
    for (int64_t i = 0; i < elementsV; i++)
        V[i] = (((floatTypeV) (rand() / RAND_MAX)) - 0.5) * 100;
    
    memset(S, 0, sizeof(floatTypeS) * elementsS);  
    memset(O, 0, sizeof(floatTypeO) * elementsO);  



    /*************************
     * nvplTENSOR
     *************************/

    /*************************
     * Create nvplTENSOR handle
     *************************/

    nvpltensorHandle_t handle;
    HANDLE_ERROR(nvpltensorCreate(&handle));

    /**********************
     * Set numbers of threads that nvplTensor can use
     **********************/
    HANDLE_ERROR(nvpltensorSetNumThreads(handle, numThreads));

    /**********************
     * Create Tensor Descriptors
     **********************/

    nvpltensorTensorDescriptor_t descQ;
    HANDLE_ERROR(nvpltensorCreateTensorDescriptor(handle, &descQ, nmodeQ, extentQ, NULL, /*stride*/
                                                  typeQ, kAlignment));

    nvpltensorTensorDescriptor_t descK;
    HANDLE_ERROR(nvpltensorCreateTensorDescriptor(handle, &descK, nmodeK, extentK, NULL, /*stride*/
                                                  typeK, kAlignment));

    nvpltensorTensorDescriptor_t descS;
    HANDLE_ERROR(nvpltensorCreateTensorDescriptor(handle, &descS, nmodeS, extentS, NULL, /*stride*/
                                                  typeS, kAlignment));

    nvpltensorTensorDescriptor_t descV;
    HANDLE_ERROR(nvpltensorCreateTensorDescriptor(handle, &descV, nmodeV, extentV, NULL, /*stride*/
                                                  typeV, kAlignment));

    nvpltensorTensorDescriptor_t descO;
    HANDLE_ERROR(nvpltensorCreateTensorDescriptor(handle, &descO, nmodeO, extentO, NULL, /*stride*/
                                                  typeO, kAlignment));                                              

    /*******************************
     * Create Contraction Descriptor
     *******************************/

    nvpltensorOperationDescriptor_t desc;
    HANDLE_ERROR(nvpltensorCreateContraction(handle, &desc, descQ, modeQ, /* unary operator A*/ NVPLTENSOR_OP_IDENTITY,
                                             descK, modeK, /* unary operator B*/ NVPLTENSOR_OP_IDENTITY, descS, modeS,
                                             /* unary operator C*/ NVPLTENSOR_OP_IDENTITY, descS, modeS, descCompute));

    /*****************************
     * Optional (but recommended): ensure that the scalar type is correct.
     *****************************/

    nvpltensorDataType_t scalarType;
    HANDLE_ERROR(nvpltensorOperationDescriptorGetAttribute(handle, desc, NVPLTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE,
                                                           (void*) &scalarType, sizeof(scalarType)));

    assert(scalarType == NVPLTENSOR_R_32F);

    /**************************
     * Set the algorithm to use
     ***************************/

    nvpltensorAlgo_t const algo = NVPLTENSOR_ALGO_DEFAULT;

    nvpltensorPlanPreference_t planPref;
    HANDLE_ERROR(nvpltensorCreatePlanPreference(handle, &planPref, algo, NVPLTENSOR_JIT_MODE_NONE));

    /**********************
     * Query workspace estimate
     **********************/

    uint64_t workspaceSizeEstimate = 0;
    nvpltensorWorksizePreference_t const workspacePref = NVPLTENSOR_WORKSPACE_DEFAULT;
    HANDLE_ERROR(nvpltensorEstimateWorkspaceSize(handle, desc, planPref, workspacePref, &workspaceSizeEstimate));

    /**************************
     * Create Contraction Plan
     **************************/

    nvpltensorPlan_t plan;
    HANDLE_ERROR(nvpltensorCreatePlan(handle, &plan, desc, planPref, workspaceSizeEstimate));

    /**************************
     * Optional: Query information about the created plan
     **************************/

    // query actually used workspace
    uint64_t actualWorkspaceSize = 0;
    HANDLE_ERROR(nvpltensorPlanGetAttribute(handle, plan, NVPLTENSOR_PLAN_REQUIRED_WORKSPACE, &actualWorkspaceSize,
                                            sizeof(actualWorkspaceSize)));

    // At this point the user knows exactly how much memory is need by the operation and
    // only the smaller actual workspace needs to be allocated
    assert(actualWorkspaceSize <= workspaceSizeEstimate);
    actualWorkspaceSize += 256;

    void* work = NULL;
    if (actualWorkspaceSize > 0)
    {
        work = aligned_alloc(kAlignment, actualWorkspaceSize);
    }

    /**********************
     * Execute first contraction and softmax and destroy plan and operator descriptor so we can make a new one
     **********************/

    HANDLE_ERROR(
        nvpltensorContract(handle, plan, (void*) &alpha, Q, K, (void*) &beta, S, S, work, actualWorkspaceSize));
    
    HANDLE_ERROR(nvpltensorDestroy(handle));
    HANDLE_ERROR(nvpltensorDestroyPlan(plan));
    HANDLE_ERROR(nvpltensorDestroyOperationDescriptor(desc));

    // Softmax
    softmax(S, extent[0], extent[1], extent[2], extent[4]);

    /*************************
     * Create nvplTENSOR handle (again)
     *************************/

    nvpltensorHandle_t handle;
    HANDLE_ERROR(nvpltensorCreate(&handle));

    /**********************
     * Set numbers of threads that nvplTensor can use (again)
     **********************/
    HANDLE_ERROR(nvpltensorSetNumThreads(handle, numThreads));

    // We skipped the tensor descriptors since we already established one on top                           

    /*******************************
     * Create Contraction Descriptor
     *******************************/

    nvpltensorOperationDescriptor_t desc;
    HANDLE_ERROR(nvpltensorCreateContraction(handle, &desc, descS, modeS, /* unary operator A*/ NVPLTENSOR_OP_IDENTITY,
                                             descV, modeV, /* unary operator B*/ NVPLTENSOR_OP_IDENTITY, descO, modeO,
                                             /* unary operator C*/ NVPLTENSOR_OP_IDENTITY, descO, modeO, descCompute));

    /*****************************
     * Optional (but recommended): ensure that the scalar type is correct.
     *****************************/

    nvpltensorDataType_t scalarType;
    HANDLE_ERROR(nvpltensorOperationDescriptorGetAttribute(handle, desc, NVPLTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE,
                                                           (void*) &scalarType, sizeof(scalarType)));

    assert(scalarType == NVPLTENSOR_R_32F);

    /**************************
     * Set the algorithm to use
     ***************************/

    nvpltensorAlgo_t const algo = NVPLTENSOR_ALGO_DEFAULT;

    nvpltensorPlanPreference_t planPref;
    HANDLE_ERROR(nvpltensorCreatePlanPreference(handle, &planPref, algo, NVPLTENSOR_JIT_MODE_NONE));

    /**********************
     * Query workspace estimate
     **********************/

    uint64_t workspaceSizeEstimate = 0;
    nvpltensorWorksizePreference_t const workspacePref = NVPLTENSOR_WORKSPACE_DEFAULT;
    HANDLE_ERROR(nvpltensorEstimateWorkspaceSize(handle, desc, planPref, workspacePref, &workspaceSizeEstimate));

    /**************************
     * Create Contraction Plan
     **************************/

    nvpltensorPlan_t plan;
    HANDLE_ERROR(nvpltensorCreatePlan(handle, &plan, desc, planPref, workspaceSizeEstimate));

    /**************************
     * Optional: Query information about the created plan
     **************************/

    // query actually used workspace
    uint64_t actualWorkspaceSize = 0;
    HANDLE_ERROR(nvpltensorPlanGetAttribute(handle, plan, NVPLTENSOR_PLAN_REQUIRED_WORKSPACE, &actualWorkspaceSize,
                                            sizeof(actualWorkspaceSize)));

    // At this point the user knows exactly how much memory is need by the operation and
    // only the smaller actual workspace needs to be allocated
    assert(actualWorkspaceSize <= workspaceSizeEstimate);
    actualWorkspaceSize += 256;

    void* work = NULL;
    if (actualWorkspaceSize > 0)
    {
        work = aligned_alloc(kAlignment, actualWorkspaceSize);
    }

    /**********************
     * Execute first contraction and softmax and destroy plan and operator descriptor so we can make a new one
     **********************/

    HANDLE_ERROR(
        nvpltensorContract(handle, plan, (void*) &alpha, S, V, (void*) &beta, O, O, work, actualWorkspaceSize));
    
    // End timer
	double elapsed_time;
	elapsed_time = (clock() - time_start)/CLOCKS_PER_SEC;
    long peak_mem = get_peak_memory_kb();  // Memory measurement

    /*************************/

    HANDLE_ERROR(nvpltensorDestroy(handle));
    HANDLE_ERROR(nvpltensorDestroyPlan(plan));
    HANDLE_ERROR(nvpltensorDestroyOperationDescriptor(desc));
    HANDLE_ERROR(nvpltensorDestroyTensorDescriptor(descQ));
    HANDLE_ERROR(nvpltensorDestroyTensorDescriptor(descK));
    HANDLE_ERROR(nvpltensorDestroyTensorDescriptor(descS));
    HANDLE_ERROR(nvpltensorDestroyTensorDescriptor(descV));
    HANDLE_ERROR(nvpltensorDestroyTensorDescriptor(descO));


    if (Q)
        free(Q);
    if (K)
        free(K);
    if (S)
        free(S);
    if (V)
        free(V);
    if (O)
        free(O);
    if (work)
        free(work);

    // Calculating flops and outputting values
    int64_t flops_qk = 2LL * extent[0] * extent[1] * extent[2] * extent[3] * extent[4];
    double gflops = (double)flops_qk / 1e9 / elapsed_time;

    printf("Simulation finished in: %f\n", elapsed_time);
    printf("Performance: %.2f GFLOPs\n", gflops);   
	printf("Peak memory usage: %ld KB (%.2f MB)\n", peak_mem, peak_mem / 1024.0);
    return 0;
}