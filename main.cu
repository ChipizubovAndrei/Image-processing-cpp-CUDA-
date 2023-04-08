#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <stdio.h>
#include <stdint.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#include "lib/stb/stb_image.h"
#include "lib/stb/stb_image_write.h"

#define N_CHANNELS 3
#define MAX_BLOCK_PER_GRID 65534

#include "include/functions.hpp"

int main()
{

  char path_to_image [] = "../images/airplane2.jpg";
  char path_to_save [] = "../output/output2.jpg";

  int count;
  cudaDeviceProp prop;

  cudaGetDeviceCount( &count );

  for (int i = 0; i < count; i++)
  {
    cudaGetDeviceProperties( &prop, i );
  }

  cudaEvent_t start, stop;
  float gpuTime = 0.0;
  int n_iter = 10;

  uint8_t* src;
  uint8_t* dst;

  uint8_t* dev_src;
  uint8_t* dev_dst;
  uint8_t* dev_kernel;
  uint8_t* dev_transfer_out;

  int width, height, ch;
  uint8_t kernel [81] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 
                        1, 1, 1, 1, 1, 1, 1, 1, 1, 
                        1, 1, 1, 1, 1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1, 1, 1, 1, 1};

  src = stbi_load( path_to_image, &width, &height, &ch, 3 );

  int N = width*height*N_CHANNELS;
  dst = (uint8_t*)malloc(N * sizeof( uint8_t ));

  cudaEventCreate( &start );
  cudaEventCreate( &stop );
  cudaEventRecord( start, 0 );


  // printf("\nCUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

  for (int i = 0; i < n_iter; i++)
  {
    cudaMalloc((void**)&dev_src, N * sizeof( uint8_t ));
    cudaMalloc((void**)&dev_dst, N * sizeof( uint8_t ));
    cudaMalloc((void**)&dev_transfer_out, N * sizeof( uint8_t ));
    cudaMalloc((void**)&dev_kernel, sizeof( kernel ));

    cudaMemcpy( dev_src, src, N * sizeof(uint8_t), cudaMemcpyHostToDevice );
    cudaMemcpy( dev_kernel, kernel, sizeof(kernel), cudaMemcpyHostToDevice );

    int threadsPerBlock = prop.maxThreadsPerBlock;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    if (blocksPerGrid > MAX_BLOCK_PER_GRID)
    {
      blocksPerGrid = MAX_BLOCK_PER_GRID;
    }

    transfer<<< blocksPerGrid, threadsPerBlock >>>( dev_src, dev_transfer_out, 
                                                  height, width, 2000, 2000 );
  
    Convolution2D<<< blocksPerGrid, threadsPerBlock >>>( dev_transfer_out, dev_dst, height, width, 
                                                        dev_kernel, 9 );

    cudaMemcpy( dst, dev_dst, N * sizeof(uint8_t), cudaMemcpyDeviceToHost );

    cudaFree( dev_src );
    cudaFree( dev_dst );
    cudaFree( dev_transfer_out );
    cudaFree( dev_kernel );

  }

  cudaEventRecord( stop, 0 );
  cudaEventSynchronize( stop );
  cudaEventElapsedTime( &gpuTime, start, stop );

  cudaEventDestroy( start );
  cudaEventDestroy( stop );

  printf( "\nTime for %d X %d X 3 image = %f milliseconds\n", width, height, gpuTime / n_iter );

  int err = stbi_write_jpg( path_to_save, width, height, ch, dst, width*ch );

  if (err == 0)
  {
    printf("\nWrite/Read error\n");
  }

  stbi_image_free( src );
  free( dst );

  return 0;
}