#include "util.h"
#include "aes_128.h"

#define KRED  "\x1B[31m"
#define KGRN  "\x1B[32m"
#define KYEL  "\x1B[33m"
#define KBLU  "\x1B[34m"
#define RESET "\x1B[0m"

#define N (33*1024)

__global__ void add( int *a, int *b, int *c ) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < N) {
        c[tid] = a[tid] + b[tid];
        tid += blockDim.x * gridDim.x;
    }

}

void test_util(void) {
    DEVICE_INFO;
    int a[N], b[N], c[N];
    int *dev_a, *dev_b, *dev_c;

    // allocate the memory on the GPU
    PERROR( cudaMalloc( (void**)&dev_a, N * sizeof(int) ) );
    PERROR( cudaMalloc( (void**)&dev_b, N * sizeof(int) ) );
    PERROR( cudaMalloc( (void**)&dev_c, N * sizeof(int) ) );

    // fill the arrays 'a' and 'b' on the CPU
    for (int i = 0; i < N; i++) {
        a[i] = i;
        b[i] = i * i;
    }

    // copy the arrays 'a' and 'b' to the GPU
    PERROR( copy_to_dev( dev_a, a, N * sizeof(int)) );
    PERROR( copy_to_dev( dev_b, b, N * sizeof(int)) );

    add<<<128,128>>>( dev_a, dev_b, dev_c );

    // copy the array 'c' back from the GPU to the CPU
    PERROR( copy_to_host( c, dev_c, N * sizeof(int)) );

    // verify the result
    for (int i = 0; i < N; i++) {
        if (a[i]+b[i]!=c[i]) {
            printf(KRED "Test failed.\n" RESET);
            goto done_free;
        }
    }
    printf(KGRN "Test passed.\n" RESET);

done_free:
    // free the memory allocated on the GPU
    PERROR( cudaFree( dev_a ) );
    PERROR( cudaFree( dev_b ) );
    PERROR( cudaFree( dev_c ) );
}

int main(void) {
    test_util();
    return 0;
}
