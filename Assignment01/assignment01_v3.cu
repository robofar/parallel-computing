#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
#include "StopWatch.h"

// Device code
// Kernel function to add two vectors
__global__ void addVectors(double* result, double* vector1, double* vector2, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        result[i] = vector1[i] + vector2[i];
    }
}


// Host code
int main() {
    std::cout << "Settings used: " << std::endl;
    std::cout << "1) Memory Used: Host and Device" << std::endl;
    std::cout << "2) Initialization of the vectors: CPU" << std::endl;
    std::cout << "3) Number of Threads per Block: int (64)" << std::endl;
    std::cout << "4) Number of Blocks per Grid: int " << std::endl;
    std::cout << std::endl;

    StopWatch execution_timer;
    execution_timer.start();

    StopWatch timer;

    // int size = 900000000;
    int size = 9000;
    size_t bytes = size * sizeof(double);
    
    // Allocate memory for the vectors on the host
    timer.start();
    double* hostVector1 = (double*)malloc(bytes);
    double* hostVector2 = (double*)malloc(bytes);
    double* hostResultCPU = (double*)malloc(bytes);
    double* hostResultGPU = (double*)malloc(bytes);
    std::cout << "Time taken for memory allocation on CPU: " << timer.elapsedTime() << " seconds" << std::endl;

    // Initialize random seed
    srand(time(NULL));
    
    // Initialize the vectors with random values on the host
    timer.start();
    for (int i = 0; i < size; i++) {
        hostVector1[i] = (double)rand() / RAND_MAX;
        hostVector2[i] = (double)rand() / RAND_MAX;
    }
    std::cout << "Time taken for vectors initialization on CPU: " << timer.elapsedTime() << " seconds" << std::endl;
    
    
    // Allocate memory for the vectors on the device
    timer.start();
    double* deviceVector1;
    double* deviceVector2;
    double* deviceResult;
    cudaMalloc(&deviceVector1, bytes);
    cudaMalloc(&deviceVector2, bytes);
    cudaMalloc(&deviceResult, bytes);
    std::cout << "Time taken for memory allocation on GPU: " << timer.elapsedTime() << " seconds" << std::endl;
    
    // Copy the vectors from host to device
    timer.start();
    cudaMemcpy(deviceVector1, hostVector1, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceVector2, hostVector2, bytes, cudaMemcpyHostToDevice);
    auto bw_host_to_device = (2 * bytes) / timer.elapsedTime(); // bytes / s (can be also converted to KB/s, MB/s or GB/s)
    std::cout << "Time taken for copying vectors from host to device: " << timer.elapsedTime() << " seconds" << std::endl;
    std::cout << "Bandwidth of the data flow from Host to Device (for copying 2 vectors): " << bw_host_to_device << " [bytes/s]" << std::endl;

    // Define kernel configuration
    int threadsPerBlock = 64;
    int blocksPerGrid = int((size + threadsPerBlock - 1) / threadsPerBlock);

    std::cout << std::endl;
    std::cout << "threadsPerBlock: " << threadsPerBlock << std::endl;
    std::cout << "blocksPerGrid: " << blocksPerGrid << std::endl;
    std::cout << std::endl;
    
    
    // Perform vector addition on the GPU
    timer.start();
    addVectors<<<blocksPerGrid, threadsPerBlock>>>(deviceResult, deviceVector1, deviceVector2, size);
    cudaDeviceSynchronize();  // Ensure that the kernel is finished before stopping the timer (read about this !!!!!!!!!!)
    std::cout << "Time taken for performing vector addition on GPU: " << timer.elapsedTime() << " seconds" << std::endl;
    
    
    // Copy the result from device to host
    // This is neccessary because we are now in host code, and therefore we cannot access device memory (remember they are separated)
    // Only threads can access device memory (i.e. you can access device memory only from device code (i.e. kernel))
    timer.start();
    cudaMemcpy(hostResultGPU, deviceResult, bytes, cudaMemcpyDeviceToHost);
    auto bw_device_to_host = bytes / timer.elapsedTime(); // bytes / s (can be also converted to KB/s, MB/s or GB/s)
    std::cout << "Time taken for copying resulting vector from device to host: " << timer.elapsedTime() << " seconds" << std::endl;
    std::cout << "Bandwidth of the data flow from Device to Host (for copying 1 vector): " << bw_device_to_host << " [bytes/s]" << std::endl;

    // Perform vector addition on the CPU
    timer.start();
    for (int i = 0; i < size; i++) {
        hostResultCPU[i] = hostVector1[i] + hostVector2[i];
    }
    std::cout << "Time taken for performing vector addition on CPU: " << timer.elapsedTime() << " seconds" << std::endl;
    std::cout << std::endl;


    // Compare the results
    std::cout << "Comparing results of CPU and GPU vector addition: " << std::endl;
    bool isEqual = true;
    for (int i = 0; i < size; i++) {
        if (hostResultCPU[i] != hostResultGPU[i]) {
            isEqual = false;
            break;
        }
    }

    if (isEqual) {
        std::cout << "Results are equal." << std::endl;
    } else {
        std::cout << "Results are not equal." << std::endl;
    }

    std::cout << std::endl;
    
    // Free device memory
    timer.start();
    cudaFree(deviceVector1);
    cudaFree(deviceVector2);
    cudaFree(deviceResult);
    std::cout << "Time taken for memory deallocation of vectors on GPU: " << timer.elapsedTime() << " seconds" << std::endl;

    
    // Free host memory
    timer.start();
    free(hostVector1);
    free(hostVector2);
    free(hostResultCPU);
    free(hostResultGPU);
    std::cout << "Time taken for memory deallocation of vectors on CPU: " << timer.elapsedTime() << " seconds" << std::endl;

    std::cout << "Execution time of whole program: " << execution_timer.elapsedTime() << " seconds" << std::endl;

    return 0;
}