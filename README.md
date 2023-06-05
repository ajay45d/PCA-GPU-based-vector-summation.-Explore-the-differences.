# PCA-EXP-1-PCA-GPU-Based-Vector-Summation

i) Using the program sumArraysOnGPU-timer.cu, set the block.x = 1023. Recompile and run it. Compare the result with the execution configuration of block.x = 1024. Try to explain the difference and the reason.

ii) Refer to sumArraysOnGPU-timer.cu, and let block.x = 256. Make a new kernel to let each thread handle two elements. Compare the results with other execution configurations.

## AIM:

To perform GPU based vector summation and explore the differences with different block values.

## PROCEDURE:

* Open "sumArraysOnGPU-timer.cu" in a text editor or IDE.

* Set "block.x" to 1023 and recompile the program. Then run it.

* Set "block.x" to 1024 and recompile the program. Then run it.

* Compare the results and observe any differences in performance.

* Set "block.x" to 256 and modify the kernel function to let each thread handle two elements.

* Recompile and run the program.

* Compare the results with other execution configurations, such as "block.x = 512" or "block.x = 1024".

* Analyze the results and observe any differences in performance.

* Repeat the steps with different input arrays and execution configurations to further explore the program's performance characteristics.

## (i) Using the program sumArraysOnGPU-timer.cu, set the block.x = 1023. Recompile and run it. Compare the result with the execution configuration of block.x = 1024. Try to explain the difference and the reason.
## PROGRAM:
common.h:
```
#include <sys/time.h>
#ifndef _COMMON_H
#define _COMMON_H
#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
	fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
	fprintf(stderr, "code: %d, reason: %s\n", error,                       \
		cudaGetErrorString(error));                                    \
	exit(1);                                                               \
    }                                                                          \
}
#define CHECK_CUBLAS(call)                                                     \
{                                                                              \
    cublasStatus_t err;                                                        \
    if ((err = (call)) != CUBLAS_STATUS_SUCCESS)                               \
    {                                                                          \
	fprintf(stderr, "Got CUBLAS error %d at %s:%d\n", err, __FILE__,       \
		__LINE__);                                                     \
	exit(1);                                                               \
    }                                                                          \
}
#define CHECK_CURAND(call)                                                     \
{                                                                              \
    curandStatus_t err;                                                        \
    if ((err = (call)) != CURAND_STATUS_SUCCESS)                               \
    {                                                                          \
	fprintf(stderr, "Got CURAND error %d at %s:%d\n", err, __FILE__,       \
		__LINE__);                                                     \
	exit(1);                                                               \
    }                                                                          \
}

#define CHECK_CUFFT(call)                                                      \
{                                                                              \
    cufftResult err;                                                           \
    if ( (err = (call)) != CUFFT_SUCCESS)                                      \
    {                                                                          \
	fprintf(stderr, "Got CUFFT error %d at %s:%d\n", err, __FILE__,        \
		__LINE__);                                                     \
	exit(1);                                                               \
    }                                                                          \
}
#define CHECK_CUSPARSE(call)                                                   \
{                                                                              \
    cusparseStatus_t err;                                                      \
    if ((err = (call)) != CUSPARSE_STATUS_SUCCESS)                             \
    {                                                                          \
	fprintf(stderr, "Got error %d at %s:%d\n", err, __FILE__, __LINE__);   \
	cudaError_t cuda_err = cudaGetLastError();                             \
	if (cuda_err != cudaSuccess)                                           \
	{                                                                      \
	    fprintf(stderr, "  CUDA error \"%s\" also detected\n",             \
		    cudaGetErrorString(cuda_err));                             \
	}                                                                      \
	exit(1);                                                               \
    }                                                                          \
}
inline double seconds()
{
    struct timeval tp;
    struct timezone tzp;
    int i = gettimeofday(&tp, &tzp);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}
#endif // _COMMON_H
```     
sumArraysOnGPU-timer.cu
```
#include "common.h"
#include <cuda_runtime.h>
#include <stdio.h>

void checkResult(float *hostRef, float *gpuRef, const int N)
{
    double epsilon = 1.0E-8;
    bool match = 1;

    for (int i = 0; i < N; i++)
    {
	if (abs(hostRef[i] - gpuRef[i]) > epsilon)
	{
	    match = 0;
	    printf("Arrays do not match!\n");
	    printf("host %5.2f gpu %5.2f at current %d\n", hostRef[i],
		   gpuRef[i], i);
	    break;
	}
    }

    if (match) printf("Arrays match.\n\n");

    return;
}

void initialData(float *ip, int size)
{
    // generate different seed for random number
    time_t t;
    srand((unsigned) time(&t));

    for (int i = 0; i < size; i++)
    {
	ip[i] = (float)( rand() & 0xFF ) / 10.0f;
    }

    return;
}

void sumArraysOnHost(float *A, float *B, float *C, const int N)
{
    for (int idx = 0; idx < N; idx++)
    {
	C[idx] = A[idx] + B[idx];
    }
}
__global__ void sumArraysOnGPU(float *A, float *B, float *C, const int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) C[i] = A[i] + B[i];
}

int main(int argc, char **argv)
{
    printf("%s Starting...\n", argv[0]);

    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    // set up data size of vectors
    int nElem = 1 << 24;
    printf("Vector size %d\n", nElem);

    // malloc host memory
    size_t nBytes = nElem * sizeof(float);

    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A     = (float *)malloc(nBytes);
    h_B     = (float *)malloc(nBytes);
    hostRef = (float *)malloc(nBytes);
    gpuRef  = (float *)malloc(nBytes);

    double iStart, iElaps;

    // initialize data at host side
    iStart = seconds();
    initialData(h_A, nElem);
    initialData(h_B, nElem);
    iElaps = seconds() - iStart;
    printf("initialData Time elapsed %f sec\n", iElaps);
    memset(hostRef, 0, nBytes);
    memset(gpuRef,  0, nBytes);

    // add vector at host side for result checks
    iStart = seconds();
    sumArraysOnHost(h_A, h_B, hostRef, nElem);
    iElaps = seconds() - iStart;
    printf("sumArraysOnHost Time elapsed %f sec\n", iElaps);

    // malloc device global memory
    float *d_A, *d_B, *d_C;
    CHECK(cudaMalloc((float**)&d_A, nBytes));
    CHECK(cudaMalloc((float**)&d_B, nBytes));
    CHECK(cudaMalloc((float**)&d_C, nBytes));

    // transfer data from host to device
    CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_C, gpuRef, nBytes, cudaMemcpyHostToDevice));

    // invoke kernel at host side
    int iLen = 512;
    dim3 block (iLen);
    dim3 grid  ((nElem + block.x - 1) / block.x);

    iStart = seconds();
    sumArraysOnGPU<<<grid, block>>>(d_A, d_B, d_C, nElem);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    printf("sumArraysOnGPU <<<  %d, %d  >>>  Time elapsed %f sec\n", grid.x,
	   block.x, iElaps);

    // check kernel error
    CHECK(cudaGetLastError()) ;

    // copy kernel result back to host side
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    // check device results
    checkResult(hostRef, gpuRef, nElem);

    // free device global memory
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));

    // free host memory
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    return(0);
}
```
## Output:
Let the block.x = 1023:
```
root@SAV-MLSystem:/home/student/Sidd_Lab_Exp_1# nvcc sumArraysOnGPU-timer.cu -o sumArraysOnGPU-timer
root@SAV-MLSystem:/home/student/Sidd_Lab_Exp_1# nvcc sumArraysOnGPU-timer.cu
root@SAV-MLSystem:/home/student/Sidd_Lab_Exp_1# ./sumArraysOnGPU-timer
./sumArraysOnGPU-timer Starting...
Using Device 0: NVIDIA GeForce GT 710
Vector size 16777216
initialData Time elapsed 0.427707 sec
sumArraysOnHost Time elapsed 0.034538 sec
sumArraysOnGPU <<<  16401, 1023  >>>  Time elapsed 0.020212 sec
Arrays match.

root@SAV-MLSystem:/home/student/Sidd_Lab_Exp_1# 
```
Let the block.x = 1024:
```
root@SAV-MLSystem:/home/student/Sidd_Lab_Exp_1# nvcc sumArraysOnGPU-timer.cu -o sumArraysOnGPU-timer
root@SAV-MLSystem:/home/student/Sidd_Lab_Exp_1# nvcc sumArraysOnGPU-timer.cu
root@SAV-MLSystem:/home/student/Sidd_Lab_Exp_1# ./sumArraysOnGPU-timer
./sumArraysOnGPU-timer Starting...
Using Device 0: NVIDIA GeForce GT 710
Vector size 16777216
initialData Time elapsed 0.423519 sec
sumArraysOnHost Time elapsed 0.034505 sec
sumArraysOnGPU <<<  16384, 1024  >>>  Time elapsed 0.020785 sec
Arrays match.

root@SAV-MLSystem:/home/student/Sidd_Lab_Exp_1# 
```
## Differences and the Reason:
* The difference between the execution configurations with block.x = 1023 and block.x = 1024 is the number of threads per block. When block.x = 1023, there are a total of 16,694 threads (16401 blocks x 1023 threads per block), while with block.x = 1024, there are 16,777,216 threads (16384 blocks x 1024 threads per block). This means that with block.x = 1024, the total number of threads is much higher than with block.x = 1023.
* In terms of performance, the difference between these two configurations is not significant. The execution time of the kernel function sumArraysOnGPU is slightly faster with block.x = 1023 (0.020212 sec) compared to block.x = 1024 (0.020785 sec), but the difference is small.
* The reason why the performance difference is small is because the number of threads that can be run in parallel is limited by the hardware resources of the GPU. In this case, the number of threads per block is already high enough that adding more threads does not result in a significant improvement in performance. Additionally, with block.x = 1024, there are fewer blocks, which can lead to underutilization of the GPU's resources.
* Overall, the performance difference between these two configurations is not significant, and the optimal configuration may depend on the specific hardware and workload.


## (ii) Refer to sumArraysOnGPU-timer.cu, and let block.x = 256. Make a new kernel to let each thread handle two elements. Compare the results with other execution configurations.
## PROGRAM:
sumArraysOnGPU-timer.cu
```
#include "common.h"
#include <cuda_runtime.h>
#include <stdio.h>
void checkResult(float *hostRef, float *gpuRef, const int N)
{
    double epsilon = 1.0E-8;
    bool match = 1;
    for (int i = 0; i < N; i++)
    {
	if (abs(hostRef[i] - gpuRef[i]) > epsilon)
	{
	    match = 0;
	    printf("Arrays do not match!\n");
	    printf("host %5.2f gpu %5.2f at current %d\n", hostRef[i],
		   gpuRef[i], i);
	    break;
	}
    }
    if (match) printf("Arrays match.\n\n");
    return;
}
void initialData(float *ip, int size)
{
    // generate different seed for random number
    time_t t;
    srand((unsigned) time(&t));
    for (int i = 0; i < size; i++)
    {
	ip[i] = (float)( rand() & 0xFF ) / 10.0f;
    }
    return;
}
void sumArraysOnHost(float *A, float *B, float *C, const int N)
{
    for (int idx = 0; idx < N; idx++)
    {
	C[idx] = A[idx] + B[idx];
    }
}
__global__ void sumArraysOnGPU(float *A, float *B, float *C, const int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}
__global__ void sumArraysOnGPU_2(float *A, float *B, float *C, const int N)
{
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (i < N) {
	C[i]   = A[i]   + B[i];
	C[i+1] = A[i+1] + B[i+1];
    }
}
int main(int argc, char **argv)
{
    printf("%s Starting...\n", argv[0]);
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));
    // set up data size of vectors
    int nElem = 1 << 24;
    printf("Vector size %d\n", nElem);
    // malloc host memory
    size_t nBytes = nElem * sizeof(float);
    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A     = (float *)malloc(nBytes);
    h_B     = (float *)malloc(nBytes);
    hostRef = (float *)malloc(nBytes);
    gpuRef  = (float *)malloc(nBytes);
    double iStart, iElaps;
    // initialize data at host side
    iStart = seconds();
    initialData(h_A, nElem);
    initialData(h_B, nElem);
    iElaps = seconds() - iStart;
    printf("initialData Time elapsed %f sec\n", iElaps);
    memset(hostRef, 0, nBytes);
    memset(gpuRef,  0, nBytes);
    // add vector at host side for result checks
    iStart = seconds();
    sumArraysOnHost(h_A, h_B, hostRef, nElem);
    iElaps = seconds() - iStart;
    printf("sumArraysOnHost Time elapsed %f sec\n", iElaps);
    // malloc device global memory
    float *d_A, *d_B, *d_C;
    CHECK(cudaMalloc((float**)&d_A, nBytes));
    CHECK(cudaMalloc((float**)&d_B, nBytes));
    CHECK(cudaMalloc((float**)&d_C, nBytes));
    // transfer data from host to device
    CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_C, gpuRef, nBytes, cudaMemcpyHostToDevice));
    // invoke kernel at host side
    // invoke kernel at host side
    int iLen = 128;
   dim3 block(iLen);
   dim3 grid((nElem / 2 + block.x - 1) / block.x);
   iStart = seconds();
   sumArraysOnGPU_2<<<grid, block>>>(d_A, d_B, d_C, nElem);
   CHECK(cudaDeviceSynchronize());
   iElaps = seconds() - iStart;
   printf("sumArraysOnGPU_2 <<< %d, %d >>> Time elapsed %f sec\n", grid.x, block.x, iElaps);   
    // check kernel error
    CHECK(cudaGetLastError()) ;
    // copy kernel result back to host side
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
    // check device results
    checkResult(hostRef, gpuRef, nElem);
    // free device global memory
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));
    // free host memory
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);
    return(0);
}
```
## Output:
Let the block.x = 128:
```
root@SAV-MLSystem:/home/student/Sidd_Lab_Exp_1# nvcc sumArraysOnGPU-timer-2.cu -o sumArraysOnGPU-timer-2
root@SAV-MLSystem:/home/student/Sidd_Lab_Exp_1# nvcc sumArraysOnGPU-timer-2.cu
root@SAV-MLSystem:/home/student/Sidd_Lab_Exp_1# ./sumArraysOnGPU-timer-2
./sumArraysOnGPU-timer-2 Starting...
Using Device 0: NVIDIA GeForce GT 710
Vector size 16777216
initialData Time elapsed 0.425350 sec
sumArraysOnHost Time elapsed 0.034510 sec
sumArraysOnGPU_2 <<< 65536, 128 >>> Time elapsed 0.020250 sec
Arrays match.

root@SAV-MLSystem:/home/student/Sidd_Lab_Exp_1# 
```
Let the block.x = 256:
```
root@SAV-MLSystem:/home/student/Sidd_Lab_Exp_1# nvcc sumArraysOnGPU-timer-2.cu -o sumArraysOnGPU-timer-2
root@SAV-MLSystem:/home/student/Sidd_Lab_Exp_1# nvcc sumArraysOnGPU-timer-2.cu
root@SAV-MLSystem:/home/student/Sidd_Lab_Exp_1# ./sumArraysOnGPU-timer-2
./sumArraysOnGPU-timer-2 Starting...
Using Device 0: NVIDIA GeForce GT 710
Vector size 16777216
initialData Time elapsed 0.425328 sec
sumArraysOnHost Time elapsed 0.034418 sec
sumArraysOnGPU_2 <<< 32768, 256 >>> Time elapsed 0.019457 sec
Arrays match.

root@SAV-MLSystem:/home/student/Sidd_Lab_Exp_1# 
```
## Differences and the Reason:
* By changing the block size from 128 to 256, the number of blocks needed to process the same amount of data was halved, from 65536 to 32768. However, since each thread now handles two elements instead of one, the total number of threads needed remains the same, which is equal to the product of the number of blocks and the block size.
* The execution time for the kernel sumArraysOnGPU-timer decreased slightly from 0.020250 sec to 0.019457 sec when the block size was changed from 128 to 256. This suggests that the optimal block size may lie between these two values.
* It is important to note that the execution time for sumArraysOnHost remains constant, as it is not affected by the block size. The overall performance of the program is determined by the execution time of the kernel sumArraysOnGPU-timer, which can be optimized by experimenting with different block sizes.


## Result:
Thus, to perform GPU based vector summation and explore the differences with different block values has been successfully performed.
