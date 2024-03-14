#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
#include "../StopWatch.h"

// Using Unified Memory instead of Host/Device Memory

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
    std::cout << "1) Memory Used: Unified" << std::endl;
    std::cout << "2) Initialization of the vectors: CPU" << std::endl;
    std::cout << "3) Number of Threads per Block: int (256)" << std::endl;
    std::cout << "4) Number of Blocks per Grid: int " << std::endl;
    std::cout << std::endl;

    StopWatch execution_timer;
    execution_timer.start();

    StopWatch timer;

    // Instead of different pointers for each: CPU and GPU separately, we can use one pointer with which we are accessing common unified memory

    // int size = 900000000;
    int size = 900000000;
    size_t bytes = size * sizeof(double);

    std::cout << "Number of elements: " << size << std::endl << std::endl;

    /*
    Pure CPU addition
    */

    // Allocate memory for the vectors on the host using cudaMallocHost (from Cuda Runtime) instead of using malloc
    timer.start();
    // double* hostVector1 = (double*)malloc(bytes);
    // double* hostVector2 = (double*)malloc(bytes);
    // double* hostResultCPU = (double*)malloc(bytes);
    // No need for hostResultGPU, since we will use Unified Memory, and we do not need to copy result from GPU to CPU in order to use that result
    // We can directly access it from CPU and GPU both using same pointer
    // double* hostResultGPU = (double*)malloc(bytes);
    double* hostVector1;
    double* hostVector2;
    double* hostResultCPU;

    auto flag_v1_cpu = cudaMallocHost(&hostVector1, bytes);
    auto flag_v2_cpu = cudaMallocHost(&hostVector2, bytes);
    auto flag_result_cpu = cudaMallocHost(&hostResultCPU, bytes);
    std::cout << "Time taken for memory allocation on CPU: " << timer.elapsedTime() << " seconds" << std::endl;

    std::cout << "Memory allocation on host for vector1: " << flag_v1_cpu << std::endl;
    std::cout << "Memory allocation on host for vector2: " << flag_v2_cpu << std::endl;
    std::cout << "Memory allocation on host for result: " << flag_result_cpu << std::endl << std::endl;

    if(flag_v1_cpu == 2 || flag_v2_cpu == 2 || flag_result_cpu == 2) {
        std::cout << "No enough memory available to allocate on the host..." << std::endl;
        std::cout << "Exiting program..." << std::endl;
        exit(0);
    }

    // Initialize random seed
    srand(time(NULL));
    
    // Initialize the vectors with random values on the host
    timer.start();
    for (int i = 0; i < size; i++) {
        hostVector1[i] = (double)rand() / RAND_MAX;
        hostVector2[i] = (double)rand() / RAND_MAX;
    }
    std::cout << "Time taken for vectors initialization on CPU: " << timer.elapsedTime() << " seconds" << std::endl;

    // Perform vector addition on the CPU
    timer.start();
    for (int i = 0; i < size; i++) {
        hostResultCPU[i] = hostVector1[i] + hostVector2[i];
    }
    std::cout << "Time taken for performing vector addition on CPU: " << timer.elapsedTime() << " seconds" << std::endl;
    std::cout << std::endl;



    /* 
    Below is Unified Memory usage. Pointers obtained with cudaMallocManaged can be used from both - CPU and GPU
    So, in the end I will compare it to pure CPU addition
    */

    // Allocate memory for the vectors in the Unified Memory Space
    timer.start();
    double* vector1;
    double* vector2;
    double* result;
    auto flag_v1 = cudaMallocManaged(&vector1, bytes);
    auto flag_v2 = cudaMallocManaged(&vector2, bytes);
    auto flag_result = cudaMallocManaged(&result, bytes);
    std::cout << "Time taken for memory allocation on Unified Memory: " << timer.elapsedTime() << " seconds" << std::endl;
    std::cout << "Memory allocation on unified memory for vector1: " << flag_v1 << std::endl;
    std::cout << "Memory allocation on unified memory for vector2: " << flag_v1 << std::endl;
    std::cout << "Memory allocation on unified memory for result: " << flag_result << std::endl << std::endl;

    if(flag_v1 == 2 || flag_v2 == 2 || flag_result == 2) {
        std::cout << "No enough memory available to allocate on the unified memory..." << std::endl;
        std::cout << "Exiting program..." << std::endl;
        exit(0);
    }
    
    // Initialize vectors (CPU or GPU initialization) - let it be CPU
    // So using Unified Memory we do not need to initialize separately CPU vectors and then copy that content to GPU vectors
    // We are just initializing one unified vector - initialization can be both on CPU and GPU since this memory can be accessed from both - CPU and GPU

    // Initialize the vectors with random values on the CPU (we can use vector1 and vector2 pointers in both CPU and GPU)
    // Since this initialization was already done on CPU, let us just copy those into here
    // No need for some copy CUDA Runtime functions, since we can use those pointers that points to managed memory on both CPU and GPU
    timer.start();
    for (int i = 0; i < size; i++) {
        vector1[i] = hostVector1[i];
        vector2[i] = hostVector2[i];
    }
    std::cout << "Time taken for vectors copying on CPU: " << timer.elapsedTime() << " seconds" << std::endl;

    // Define kernel configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = int((size + threadsPerBlock - 1) / threadsPerBlock);

    std::cout << std::endl;
    std::cout << "threadsPerBlock: " << threadsPerBlock << std::endl;
    std::cout << "blocksPerGrid: " << blocksPerGrid << std::endl;
    std::cout << std::endl;
    
    
    // Perform vector addition on the GPU
    timer.start();
    addVectors<<<blocksPerGrid, threadsPerBlock>>>(result, vector1, vector2, size);
    cudaDeviceSynchronize();  // Ensure that the kernel is finished before stopping the timer
    std::cout << "Time taken for performing vector addition on GPU: " << timer.elapsedTime() << " seconds" << std::endl;

    // Compare the results between result obtained using Pure CPU and results obtained using Unified Memory
    std::cout << "Comparing results of CPU and GPU vector addition: " << std::endl;
    bool isEqual = true;
    for (int i = 0; i < size; i++) {
        if (hostResultCPU[i] != result[i]) {
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
    
    
    // Since we are using common pointers (not separate ones for CPU and GPU) part of the exercise for comparing CPU result vector and GPU result vector
    // can not be done (so that part of the code is ommited)
    
    // Free Unified Memory
    timer.start();
    cudaFree(vector1);
    cudaFree(vector2);
    cudaFree(result);
    std::cout << "Time taken for memory deallocation of vectors on Unified Memory Space: " << timer.elapsedTime() << " seconds" << std::endl;

    // Free host memory
    timer.start();
    //free(hostVector1);
    //free(hostVector2);
    //free(hostResultCPU);
    cudaFreeHost(hostVector1);
    cudaFreeHost(hostVector2);
    cudaFreeHost(hostResultCPU);
    std::cout << "Time taken for memory deallocation of vectors on CPU: " << timer.elapsedTime() << " seconds" << std::endl;

    std::cout << "Execution time of whole program: " << execution_timer.elapsedTime() << " seconds" << std::endl;

    return 0;
}