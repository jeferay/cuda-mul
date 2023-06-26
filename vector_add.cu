#include <stdio.h>
#include <cuda.h>
#include <ctime>

#define N  100000

__global__ void add( int *a, int *b, int *c ) {
    int tid = blockIdx.x;
    if (tid < N)
        c[tid] = a[tid] + b[tid];
}
int a[N], b[N], c[N];
int main( void ) {
    
    int *dev_a, *dev_b, *dev_c;

    // allocate the memory on the GPU
    cudaMalloc( (void**)&dev_a, N * sizeof(int) );
    cudaMalloc( (void**)&dev_b, N * sizeof(int) );
    cudaMalloc( (void**)&dev_c, N * sizeof(int) );

    // fill arrays 'a' and 'b' with some numbers
    for (int i = 0; i < N; i++) {
        a[i] = -i;
        b[i] = i * i;
    }
    // copy arrays 'a' and 'b' to the GPU
    cudaMemcpy( dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice );
    cudaMemcpy( dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice );

    // repeatedly calculate until clock down
    time_t start_time = time(0);
    time_t now_time;
    while(1){
        now_time = time(0);
        if (now_time - start_time > 20){
            break;
        }
        add<<<N,1>>>( dev_a, dev_b, dev_c );
    }
    printf("AThe time of the whole process is %d s.\n",now_time-start_time);

    // copy array 'c' back from the GPU to the CPU
    cudaMemcpy( c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost );

    // free the memory allocated on the GPU
    cudaFree( dev_a );
    cudaFree( dev_b );
    cudaFree( dev_c );

    return 0;
}
