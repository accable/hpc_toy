/* This code was taken from https://github.com/NVIDIA/NVPLSamples/blob/main/nvpl_tensor/contraction/contraction.c and 
* and modified to accomodate scaled dot product Attention proposed by Vaswani et al., 2017. 
* 
*
* Since scaled dot product Attention requires Q, K^t, and V values, we need to also declare the intermediate values too
* (i.e., S = Q * K^t, the attention kernel), and the output O.
*/

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <nvpl_tensor.h>

// Since the cluster does not have perf or gperf, we need to do this the old fashioned way
#include <time.h>

// Memory profiling functions
long get_peak_memory_kb(){
	
	struct rusage usage;
	getrusage(RUSAGE_SELF, &usage);
	return usage.ru_maxrss;
}

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

    // Timekeeping purposes
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
     * O[B, H, O_len, D] (output)s
     **********************/

    floatTypeCompute alpha = (floatTypeCompute) 1.0f;  // We set it to 1.0f
    floatTypeCompute beta = (floatTypeCompute) 0.f;  // Is 0.0f so C won't be used

    int32_t modeQ[] = {0, 1, 2, 3};  // Q[B, H, Q_len, D]
    int32_t modeK[] = {0, 1, 3, 4};  // K[B, H, D, K_len]
    int32_t modeS[] = {0, 1, 2, 4};  // S[B, H, Q_len, K_len]

    enum { nmodeQ = 4 };
    enum { nmodeK = 4 };
    enum { nmodeS = 4 };

    int64_t extent[] = {1, 1, 128, 64, 128};  // [B, H, len, D, len]

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

    /**********************
     * Allocating data
     **********************/

    int64_t elementsQ = 1;
    for (int i = 0; i < nmodeQ; ++i)
    {
        elementsQ *= extent[i];
    }
    int64_t elementsK = 1;
    for (int i = 0; i < nmodeK; ++i)
    {
        elementsK *= extent[i];
    }
    int64_t elementsS = 1;
    for (int i = 0; i < nmodeS; ++i)
    {
        elementsS *= extent[i];
    }

    int64_t sizeQ = sizeof(floatTypeQ) * elementsQ;
    int64_t sizeK = sizeof(floatTypeK) * elementsK;
    int64_t sizeS = sizeof(floatTypeS) * elementsS;

    uint32_t const kAlignment = 128;  // Alignment of the pointers (bytes)

    floatTypeQ* Q = aligned_alloc(kAlignment, sizeQ);
    floatTypeK* K = aligned_alloc(kAlignment, sizeK);
    floatTypeS* S = aligned_alloc(kAlignment, sizeS);

    if (Q == NULL || K == NULL || S == NULL)
    {
        printf("Error: allocation of tensor memory.\n");
        return -1;
    }

    /*******************
     * Initialize data
     *******************/

    for (int64_t i = 0; i < elementsQ; i++)
        Q[i] = (((floatTypeQ) (rand() / RAND_MAX)) - 0.5) * 100;
    for (int64_t i = 0; i < elementsK; i++)
        K[i] = (((floatTypeK) (rand() / RAND_MAX)) - 0.5) * 100;
    
    memset(S, 0, sizeof(floatTypeS) * elementsS);  // We keep it 0 since this is used to keep values of Q * K^t




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
     * Execute
     **********************/

    HANDLE_ERROR(
        nvpltensorContract(handle, plan, (void*) &alpha, Q, K, (void*) &beta, S, S, work, actualWorkspaceSize));
    
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

    if (Q)
        free(Q);
    if (K)
        free(K);
    if (S)
        free(S);
    if (work)
        free(work);

    // Calculating flops and outputting values
    int64_t flops_qk = 2LL * 1 * 1 * 128 * 128 * 64;
    double gflops = (double)flops_qk / 1e9 / elapsed_time;

    printf("Simulation finished in: %f\n", elapsed_time);
    printf("Performance: %.2f GFLOPs\n", gflops);   
	printf("Peak memory usage: %ld KB (%.2f MB)\n", peak_mem, peak_mem / 1024.0);
    return 0;
}