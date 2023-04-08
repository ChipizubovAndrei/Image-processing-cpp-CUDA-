__global__ void transfer( uint8_t* src, uint8_t* dst, int srcH, 
                         int srcW, int stepy, int stepx );

__global__ void Convolution2D(uint8_t* src, uint8_t* dst, int srcH, int srcW,  uint8_t* kernel, int kernelYX);

void print_cuda_device_info( cudaDeviceProp &prop );

int getSPcores( cudaDeviceProp devProp );


/*
Функция переноса пикселей изображения
Аргументы:
  - src - исходное изображение
  - dst - возвращаемое изображение
  - stepy, stepx - количество пикселей на которое необходимо перенести изображение
  - num_rows - количество строк, которое необходимо обработать
  - num_thread - номер потока
  - remainder - остаток строк при не целом делении на потоки
*/
__global__ void transfer( uint8_t* src, uint8_t* dst, int srcH, 
                          int srcW, int stepy, int stepx )
{
  uint8_t nonzero [3] = {187, 38, 73};
  int tid = threadIdx.x + blockIdx.x*blockDim.x;

  while (tid < srcH*srcW*N_CHANNELS)
  {
    int y = tid / (srcW*N_CHANNELS);
    int x = (tid - (y*srcW*N_CHANNELS)) / N_CHANNELS;

    if (x >= stepx && y >= stepy)
    {
      dst[(y*srcW + x)*N_CHANNELS + 0] = src[((y - stepy)*srcW + (x - stepx))*N_CHANNELS + 0];
      dst[(y*srcW + x)*N_CHANNELS + 1] = src[((y - stepy)*srcW + (x - stepx))*N_CHANNELS + 1];
      dst[(y*srcW + x)*N_CHANNELS + 2] = src[((y - stepy)*srcW + (x - stepx))*N_CHANNELS + 2];
    }
    else
    {
      dst[(y*srcW + x)*N_CHANNELS + 0] = nonzero[0];
      dst[(y*srcW + x)*N_CHANNELS + 1] = nonzero[1];
      dst[(y*srcW + x)*N_CHANNELS + 2] = nonzero[2];
    }
    tid += blockDim.x * gridDim.x;
  }
  // __syncthreads();
}

__global__ void Convolution2D(uint8_t* src, uint8_t* dst, 
                              int srcH, int srcW,  uint8_t* kernel, 
                              int kernelYX)
{
  // printf("BlockDim.x = %d\n", blockDim.x);
  // printf("gridDim.x = %d\n", gridDim.x);

  int m_pad = kernelYX / 2;
  int dstH = (int)((srcH + 2 * m_pad - 1 * (kernelYX - 1) - 1) / 1 + 1);
  int dstW = (int)((srcW + 2 * m_pad - 1 * (kernelYX - 1) - 1) / 1 + 1);
  int tid = threadIdx.x + blockIdx.x*blockDim.x;

  while (tid < srcH*srcW*N_CHANNELS)
  {
    int y = tid / (srcW*N_CHANNELS);
    int x = (tid - (y*srcW*N_CHANNELS)) / N_CHANNELS;

    // Перебор по фильтрам (выходным каналам)
    for (int ch = 0; ch < N_CHANNELS; ch++)
    {
      // Проход по ядру
      int sum = 0;
      for (int ky = 0; ky < kernelYX; ky++)
      {
        for (int kx = 0; kx < kernelYX; kx++)
        {
          int sy = (int)(y - m_pad + ky);
          int sx = (int)(x - m_pad + kx);
          if (sy >= 0 && sy < srcH && sx >= 0 && sx < srcW)
          {
            sum += src[(sy*srcW + sx)*N_CHANNELS + ch]*
                    kernel[ky*kernelYX + kx];
          }
        }
      }
      dst[(y*dstW + x)*N_CHANNELS + ch] = (uint8_t)(sum / (kernelYX*kernelYX));
    }
    tid += blockDim.x * gridDim.x;
  }
  // __syncthreads();
}

int getSPcores( cudaDeviceProp devProp )
{
  int cores = 0;
  int mp = devProp.multiProcessorCount;
  switch (devProp.major) 
  {
    case 2: // Fermi
      if (devProp.minor == 1) cores = mp * 48;
      else cores = mp * 32;
      break;
    case 3: // Kepler
      cores = mp * 192;
      break;
    case 5: // Maxwell
      cores = mp * 128;
      break;
    case 6: // Pascal
      if (devProp.minor == 1) cores = mp * 128;
      else if (devProp.minor == 0) cores = mp * 64;
      else printf("Unknown device type\n");
      break;
    case 7: // Volta
      if (devProp.minor == 0) cores = mp * 64;
      else printf("Unknown device type\n");
      break;
    default:
      printf("Unknown device type\n");
      break;
  }
  return cores;
}

void print_cuda_device_info(cudaDeviceProp &prop)
{
  printf("Device name:                                        %s\n", prop.name);
  printf("Global memory available on device:                  %zu\n", prop.totalGlobalMem);
  printf("Shared memory available per block:                  %zu\n", prop.sharedMemPerBlock);
  printf("Count of 32-bit registers available per block:      %i\n", prop.regsPerBlock);
  printf("Warp size in threads:                               %i\n", prop.warpSize);
  printf("Maximum pitch in bytes allowed by memory copies:    %zu\n", prop.memPitch);
  printf("Maximum number of threads per block:                %i\n", prop.maxThreadsPerBlock);
  printf("Maximum size of each dimension of a block[0]:       %i\n", prop.maxThreadsDim[0]);
  printf("Maximum size of each dimension of a block[1]:       %i\n", prop.maxThreadsDim[1]);
  printf("Maximum size of each dimension of a block[2]:       %i\n", prop.maxThreadsDim[2]);
  printf("Maximum size of each dimension of a grid[0]:        %i\n", prop.maxGridSize[0]);
  printf("Maximum size of each dimension of a grid[1]:        %i\n", prop.maxGridSize[1]);
  printf("Maximum size of each dimension of a grid[2]:        %i\n", prop.maxGridSize[2]);
  printf("Clock frequency in kilohertz:                       %i\n", prop.clockRate);
  printf("totalConstMem:                                      %zu\n", prop.totalConstMem);
  printf("Major compute capability:                           %i\n", prop.major);
  printf("Minor compute capability:                           %i\n", prop.minor);
  printf("Number of multiprocessors on device:                %i\n", prop.multiProcessorCount);
  printf("Count of cores:                                     %i\n", getSPcores(prop));
}