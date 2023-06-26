#include <cuda.h>
#include <stdio.h>

#define N 1000000000
#define THREADS_PER_BLOCK 256

__global__ void square(float* d_in, float* d_out) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    float f = d_in[idx];
    for(int i = 0; i < 100000000; ++i) {
        f = f * f;
    }
    d_out[idx] = f;
}

__global__ void expensiveComputation(float* d_out) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    float sum = 0.0;
    for(long i = 0; i < 1000000; ++i) {
        sum += cosf(sum);
    }
    d_out[idx] = sum;
}



int main() {
    float *h_in, *h_out;
    float *d_in, *d_out;

    size_t size = N * sizeof(float);

    h_in = (float*)malloc(size);
    h_out = (float*)malloc(size);

    cudaMalloc((void**)&d_in, size);
    cudaMalloc((void**)&d_out, size);

    for (int i = 0; i < N; i++) {
        h_in[i] = float(i);
    }
    time_t start_time = time(0);
    time_t now_time;
    while(1){
        now_time = time(0);
        if (now_time - start_time > 20){
            break;
        }
        cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

        expensiveComputation<<<N/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_out);

        cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);
    }
    printf("cThe time of the whole process is %d s.\n",now_time-start_time);


    free(h_in);
    free(h_out);

    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}
