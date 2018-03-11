/* author: Kexuan Zou
   date: 03/01/2018
   external source: https://developer.nvidia.com/cuda-example
*/

#ifndef _UTIL_H
#define _UTIL_H
#include <stdio.h>

#define DEVICE_INFO device_info()
#define PERROR(e) (__perror(e, __FILE__, __LINE__))
#define copy_to_dev(d, h, s) (cudaMemcpy(d, h, s, cudaMemcpyHostToDevice))
#define copy_to_host(h, d, s) (cudaMemcpy(h, d, s, cudaMemcpyDeviceToHost))

#define MEM_ERR(a) \
    {if (a == NULL) { \
        printf( "Host memory failed in %s at line %d\n", __FILE__, __LINE__); \
        exit( EXIT_FAILURE );}}

/**
 * handles cuda error by printing error message and abort.
 * @param err  cudaError_t error object
 * @param file file in which error occurs
 * @param line line in which error occurs
 */
static void __perror(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}

/**
 * prints info of all devices on the host.
 */
static void device_info(void) {
    cudaDeviceProp device;
    int count, i;
    PERROR(cudaGetDeviceCount(&count));
    printf("%d devices found.\n", count);
    for (i = 0; i < count; i++) {
        PERROR(cudaGetDeviceProperties(&device, i));
        printf( "   --- General Information for device %d ---\n", i );
        printf( "Name:  %s\n", device.name );
        printf( "Compute capability:  %d.%d\n", device.major, device.minor );
        printf( "Clock rate:  %d\n", device.clockRate );
        printf( "   --- Memory Information for device %d ---\n", i );
        printf( "Total global mem:  %ld\n", device.totalGlobalMem );
        printf( "Total constant mem:  %ld\n", device.totalConstMem );
        printf( "Max mem pitch:  %ld\n", device.memPitch );
        printf( "Texture alignment:  %ld\n", device.textureAlignment );

        printf( "   --- MP Information for device %d ---\n", i );
        printf( "Multiprocessor count:  %d\n",
                    device.multiProcessorCount );
        printf( "Shared mem per mp:  %ld\n", device.sharedMemPerBlock );
        printf( "Registers per mp:  %d\n", device.regsPerBlock );
        printf( "Threads in warp:  %d\n", device.warpSize );
        printf( "Max threads per block:  %d\n",
                    device.maxThreadsPerBlock );
        printf( "Max thread dimensions:  (%d, %d, %d)\n",
                    device.maxThreadsDim[0], device.maxThreadsDim[1],
                    device.maxThreadsDim[2] );
        printf( "Max grid dimensions:  (%d, %d, %d)\n",
                    device.maxGridSize[0], device.maxGridSize[1],
                    device.maxGridSize[2] );
    }
}

#endif
