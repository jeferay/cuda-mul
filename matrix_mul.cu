#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

const int WIDTH = 4096;

__global__ void matrixMulKernel(float* d_M, float* d_N, float* d_P, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    float sum = 0;
    for (int i = 0; i < width; i++) {
        sum += d_M[row * width + i] * d_N[i * width + col];
    }
    
    d_P[row * width + col] = sum;
}

int main() {
    // detect the device
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess) {
        printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
        return 1;
    }

    // This function call returns 0 if there are no CUDA capable devices.
    if (deviceCount == 0) {
        printf("There are no available device(s) that support CUDA\n");
    } else {
        printf("Detected %d CUDA Capable device(s)\n", deviceCount);
    }
    int size = WIDTH * WIDTH * sizeof(float);
    
    // Allocate memory on host
    float* h_M = (float*)malloc(size);
    float* h_N = (float*)malloc(size);
    float* h_P = (float*)malloc(size);
    
    // Initialize matrices on host
    for (int i = 0; i < WIDTH * WIDTH; i++) {
        h_M[i] = rand() / (float)RAND_MAX;
        h_N[i] = rand() / (float)RAND_MAX;
    }
    
    // Allocate memory on device
    float* d_M, * d_N, * d_P;
    cudaMalloc(&d_M, size);
    cudaMalloc(&d_N, size);
    cudaMalloc(&d_P, size);
    
    // Copy matrices to device
    cudaMemcpy(d_M, h_M, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, h_N, size, cudaMemcpyHostToDevice);
    
    // Launch kernel
    dim3 dimGrid(WIDTH / 16, WIDTH / 16, 1);
    dim3 dimBlock(16, 16, 1);

    auto start = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration_ms = end - start;
    while (true){
        end = std::chrono::high_resolution_clock::now();
        duration_ms = end - start;
        if (int(duration_ms.count()) % 1000==0){
            printf("Output something everything second\n");
        }
        if (duration_ms.count()>600000){
            break;
        }
        matrixMulKernel <<< dimGrid, dimBlock >>> (d_M, d_N, d_P, WIDTH);
        cudaDeviceSynchronize();
    }

    // Copy result back to host
    cudaMemcpy(h_P, d_P, size, cudaMemcpyDeviceToHost);
    
    // Clean up
    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);
    free(h_M);
    free(h_N);
    free(h_P);

    std::cout << "Time elapsed: " << duration_ms.count() << " ms" << std::endl;

    return 0;
}
