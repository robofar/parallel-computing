#include <iostream>
#include <cstdlib>
#include <ctime>
#include <limits>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "../StopWatch.h"

// Kernel function to initialize a array with random values
__global__ void initVectorAndFindMax(float* array, int size, unsigned long long seed, float* maxVal, int* maxIdx) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    curand_init(seed, i, 0, &state);
    if (i < size) {
        array[i] = curand_uniform(&state);

        if(array[i] > *maxVal) {
            // atomicExch(ptr, newValue) 
            // swaps the content at the memory location pointed to by ptr with the new value newValue
            atomicExch(maxVal, array[i]);
            atomicExch(maxIdx, i);
        }
    }
}


// Host Code
int main() {
    StopWatch timer;

    int size = 900000000;
    size_t bytes = size * sizeof(float); // float -> 4 bytes

    /* ================= GPU ================================ */
    // 1. Allocate array on GPU
    timer.start();
    float* deviceArray;
    auto flag_device = ::cudaMalloc(&deviceArray, bytes);
    std::cout << "Time needed for memory allocation on GPU: " << timer.elapsedTime() << " seconds." << std::endl;

    if(flag_device == 2) {
        std::cout << "No enough memory available to allocate on the device..." << std::endl;
        std::cout << "Exiting program..." << std::endl;
        exit(0);
    }

    // 2. Initialize array on GPU (+ simultaneously find max index and value in same kernel)
    float* maxValGPU;
    int* maxIdxGPU;
    ::cudaMalloc(&maxValGPU, sizeof(float));
    ::cudaMalloc(&maxIdxGPU, sizeof(int));

    // Initialize maxValGPU with lowest possible float value (that will be some negative value)
    float initialVal = std::numeric_limits<float>::lowest();
    ::cudaMemcpy(maxValGPU, &initialVal, sizeof(float), cudaMemcpyHostToDevice);
    
    timer.start();
    int threadsPerBlock = 256;
    int blocksPerGrid = int((size + threadsPerBlock - 1) / threadsPerBlock);
    unsigned long long seed = time(NULL);
    initVectorAndFindMax<<<blocksPerGrid, threadsPerBlock>>>(deviceArray, size, seed, maxValGPU, maxIdxGPU);
    cudaDeviceSynchronize(); // ensures that kernel execution finishes first before moving to next sequence of command in host code
    std::cout << "Time needed for array initialization on GPU (+ max finding): " << timer.elapsedTime() << " seconds." << std::endl;

    // 3. Copy max index and max value from GPU to CPU
    float maxValGPU_host;
    int maxIdxGPU_host;
    cudaMemcpy(&maxValGPU_host, maxValGPU, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&maxIdxGPU_host, maxIdxGPU, sizeof(int), cudaMemcpyDeviceToHost);

    // 4. Print GPU max_index and max_value
    std::cout << "(maxIndexGPU, maxValueGPU): (" << maxIdxGPU_host << "," << maxValGPU_host << ")" << std::endl;


    /* ========================================================================================================= */


    /* ================= Allocate array on CPU and initialize it with content of deviceArray ================================ */

    // 5. Allocate array on CPU
    timer.start();
    float* hostArray;
    auto flag_host = ::cudaMallocHost(&hostArray, bytes);
    std::cout << "Time needed for memory allocation on CPU: " << timer.elapsedTime() << " seconds." << std::endl;

    if(flag_host == 2) {
        std::cout << "No enough memory available to allocate on the host..." << std::endl;
        std::cout << "Exiting program..." << std::endl;
        exit(0);
    }

    // 6. Copy content of GPU Array to CPU Array
    timer.start();
    auto flag_copying = cudaMemcpy(hostArray, deviceArray, bytes, cudaMemcpyDeviceToHost);
    std::cout << "Time needed for content copying from GPU array to CPU array: " << timer.elapsedTime() << " seconds." << std::endl;

    if(flag_host == 21) {
        std::cout << "Direction of the memory copying passed to the API call is not right (try sawping it)..." << std::endl;
        std::cout << "Exiting program..." << std::endl;
        exit(0);
    }

    // 7. Find max index and value on CPU
    int maxIndexCPU = 0;
    float maxValueCPU = hostArray[0];

    for (int i = 0; i < size; i++) {
        if (hostArray[i] > maxValueCPU) {
            maxValueCPU = hostArray[i];
            maxIndexCPU = i;
        }
    }

    // 8. Print CPU max_index and max_value
    std::cout << "(maxIndexCPU, maxValueCPU): (" << maxIndexCPU << "," << maxValueCPU << ")" << std::endl;


    /* ========================================================================================================= */

    // 9. Free all memory that has been allocated
    cudaFree(maxValGPU);
    cudaFree(maxIdxGPU);
    cudaFree(deviceArray);
    cudaFreeHost(hostArray);


    return 0;


}