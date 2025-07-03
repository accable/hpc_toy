## Description

This is a small collection of toy HPC problems, mainly focused as both self-exercise and practicing the use of high performance numerical libraries such as BLAS, LAPACK, and NVIDIA Performance Libraries (NVPL).

The source code is typed with experimentation in mind, thus the implementations will not be 100% correct but follows how the official NVPL sample code (https://github.com/NVIDIA/NVPLSamples) was created.

**This repository does not accept any pull requests. Any requests would be ignored.**

## Dependencies and How to Use

**Currently, this repository assumes that the user owns or have access to:**

1. Nvidia Grace Superchip w/ 240GB memory, and
2.  ```nvhpc/25.5``` module (for ```nvc``` compiler).

The code does not have any other dependencies besides NVPL.

**How to use:**

1.  Clone the repository:
    ```bash
    https://github.com/accable/hpc_toy.git
    ```

2.  Navigate to the project directory:
    ```bash
    cd hpc_toy
    cd src
    ```

3.  To compile the .c files, first load the ```nvhpc``` module:
    ```bash
    module load nvhpc/25.5
    ```
    Then compile the files using the ```nvc``` compiler:
    ```bash
    nvc 1_1_heat_diffusion_dynmem.c -o foo
    ```
    If BLAS/LAPACK is required, simply add ```-lblas``` or ```-llapack``` on the compiler arguments:
    ```bash
    nvc 1_2_heat_diffusion_dynmem_lapack.c -o foolapack -llapack
    ```
    Since some of the files require nvplTENSOR, add ```-lnvpl_tensor``` on the compiler arguments:
    ```bash
    nvc 2_0_multi_head_attention_cutensor.c -o footensor -lnvpl_tensor
    ```
    Some of the files also require specific way of compiling, add ```-Mnvpl``` on the compiler arguments:
    ```bash
    nvc 2_1_multi_head_attention_blas.c -o fooblas -Mnvpl=blas
    ```

4.  Run the application:
    ```bash
    ./foo
    ```
