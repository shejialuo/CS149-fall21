#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <driver_functions.h>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

#include "CycleTimer.h"

#define THREADS_PER_BLOCK 256

// Here, we do the operation in-place.
__global__ void scan_kernel_upsweep(int N, int two_d, int two_dplus1, int* result) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N) {
        result[(index + 1) * two_dplus1 - 1] += result[index * two_dplus1 + two_d - 1];
    }
}

// Here, we do the operation in-place.
__global__ void scan_kernel_downsweep(int N, int two_d, int two_dplus1, int* result) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N) {
        int temp = result[index * two_dplus1 + two_d - 1];
        result[index * two_dplus1 + two_d - 1] = result[(index + 1) * two_dplus1 - 1];
        result[(index + 1) * two_dplus1 - 1] += temp;
    }
}

// It may seem ugly, but it works.
__global__ void scan_set_zero_kernel(int N, int *result) {
    result[N - 1] = 0;
}

// helper function to round an integer up to the next power of 2
static inline int nextPow2(int n) {
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

// exclusive_scan --
//
// Implementation of an exclusive scan on global memory array `input`,
// with results placed in global memory `result`.
//
// N is the logical size of the input and output arrays, however
// students can assume that both the start and result arrays we
// allocated with next power-of-two sizes as described by the comments
// in cudaScan().  This is helpful, since your parallel scan
// will likely write to memory locations beyond N, but of course not
// greater than N rounded up to the next power of 2.
//
// Also, as per the comments in cudaScan(), you can implement an
// "in-place" scan, since the timing harness makes a copy of input and
// places it in result
void exclusive_scan(int N, int* result)
{

    int rounded_length = nextPow2(N);

    for(int two_d = 1; two_d <= rounded_length / 2; two_d *= 2) {
        int two_dplus1= 2 * two_d;
        // Here, we should calculate the total task number we need
        int totalTaskNum = rounded_length % two_dplus1 ? rounded_length / two_dplus1 + 1 
                                                       : rounded_length / two_dplus1;
        // Thus, we could calculate the block we need.
        int blocks = totalTaskNum % THREADS_PER_BLOCK ? totalTaskNum / THREADS_PER_BLOCK + 1
                                                      : totalTaskNum / THREADS_PER_BLOCK;
        scan_kernel_upsweep<<<blocks, THREADS_PER_BLOCK>>>(totalTaskNum, two_d, two_dplus1, result);
        cudaDeviceSynchronize();
    }

    // Here, I don't know how to find a good way
    // to set the result[N - 1] = 0. So I use
    // a simple way.
    scan_set_zero_kernel<<<1,1>>>(rounded_length, result);
    cudaDeviceSynchronize();

    for(int two_d = rounded_length / 2; two_d >= 1; two_d /= 2) {
        int two_dplus1= 2 * two_d;
        int totalTaskNum = rounded_length % two_dplus1 ? rounded_length / two_dplus1 + 1 
                                                       : rounded_length / two_dplus1;
        int blocks = totalTaskNum % THREADS_PER_BLOCK ? totalTaskNum / THREADS_PER_BLOCK + 1
                                                      : totalTaskNum / THREADS_PER_BLOCK;
        scan_kernel_downsweep<<<blocks, THREADS_PER_BLOCK>>>(totalTaskNum, two_d, two_dplus1, result);
        cudaDeviceSynchronize();
    }
}


//
// cudaScan --
//
// This function is a timing wrapper around the student's
// implementation of scan - it copies the input to the GPU
// and times the invocation of the exclusive_scan() function
// above. Students should not modify it.
double cudaScan(int* inarray, int* end, int* resultarray)
{
    int* device_result;
    int N = end - inarray;

    // This code rounds the arrays provided to exclusive_scan up
    // to a power of 2, but elements after the end of the original
    // input are left uninitialized and not checked for correctness.
    //
    // Student implementations of exclusive_scan may assume an array's
    // allocated length is a power of 2 for simplicity. This will
    // result in extra work on non-power-of-2 inputs, but it's worth
    // the simplicity of a power of two only solution.

    int rounded_length = nextPow2(end - inarray);

    cudaMalloc((void **)&device_result, sizeof(int) * rounded_length);

    // Here I choose the in-place way
    cudaMemcpy(device_result, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    exclusive_scan(N, device_result);

    // Wait for completion
    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();

    cudaMemcpy(resultarray, device_result, (end - inarray) * sizeof(int), cudaMemcpyDeviceToHost);

    double overallDuration = endTime - startTime;
    return overallDuration;
}


// cudaScanThrust --
//
// Wrapper around the Thrust library's exclusive scan function
// As above in cudaScan(), this function copies the input to the GPU
// and times only the execution of the scan itself.
//
// Students are not expected to produce implementations that achieve
// performance that is competition to the Thrust version, but it is fun to try.
double cudaScanThrust(int* inarray, int* end, int* resultarray) {

    int length = end - inarray;
    thrust::device_ptr<int> d_input = thrust::device_malloc<int>(length);
    thrust::device_ptr<int> d_output = thrust::device_malloc<int>(length);

    cudaMemcpy(d_input.get(), inarray, length * sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    thrust::exclusive_scan(d_input, d_input + length, d_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();

    cudaMemcpy(resultarray, d_output.get(), length * sizeof(int), cudaMemcpyDeviceToHost);

    thrust::device_free(d_input);
    thrust::device_free(d_output);

    double overallDuration = endTime - startTime;
    return overallDuration;
}

__global__ void find_repeats_compare(int *input, int *output, int N) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N - 1) {
        output[index] = input[index] == input[index + 1] ? 1 : 0;
    }
}

__global__ void gather_kernel(int* exclusive_scan_results, int* output, const int N, int* total_count) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N - 1) {
        if (exclusive_scan_results[index] != exclusive_scan_results[index + 1]) {
            output[exclusive_scan_results[index]] = index;
        }
    } else if (index == N - 1) {
        *total_count = exclusive_scan_results[N - 1];
    }
}

// find_repeats --
//
// Given an array of integers `device_input`, returns an array of all
// indices `i` for which `device_input[i] == device_input[i+1]`.
//
// Returns the total number of pairs found
int find_repeats(int* device_input, int length, int* device_output) {

    // The problem is that how should we use `prefix_sum` to solve
    // the `find_repeats` problem. It is tricky, you should look at
    // the code carefully.

    const int rounded_length = nextPow2(length);
    int blocks = rounded_length % THREADS_PER_BLOCK ? rounded_length / THREADS_PER_BLOCK + 1
                                                    : rounded_length / THREADS_PER_BLOCK;

    int* temp;
    cudaMalloc(&temp, rounded_length * sizeof(int));
    find_repeats_compare<<<blocks, THREADS_PER_BLOCK>>>(device_input, temp, length);

    // exclusive scan on indicator array
    // to get device_exclusive_scan_results
    exclusive_scan(length, temp);

    // get repetition points in array    
    int* device_repetition_count;
    cudaMalloc(&device_repetition_count, sizeof(int));
    gather_kernel<<<blocks, THREADS_PER_BLOCK>>>(temp, device_output, length, device_repetition_count);
    cudaFree(temp);

    // return results
    int output_length;
    cudaMemcpy(&output_length, device_repetition_count, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(device_repetition_count);
    return output_length;
}


//
// cudaFindRepeats --
//
// Timing wrapper around find_repeats. You should not modify this function.
double cudaFindRepeats(int *input, int length, int *output, int *output_length) {

    int *device_input;
    int *device_output;
    int rounded_length = nextPow2(length);

    cudaMalloc((void **)&device_input, rounded_length * sizeof(int));
    cudaMalloc((void **)&device_output, rounded_length * sizeof(int));
    cudaMemcpy(device_input, input, length * sizeof(int), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    double startTime = CycleTimer::currentSeconds();

    int result = find_repeats(device_input, length, device_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();

    // set output count and results array
    *output_length = result;
    cudaMemcpy(output, device_output, length * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(device_input);
    cudaFree(device_output);

    float duration = endTime - startTime;
    return duration;
}



void printCudaInfo()
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++)
    {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");
}
