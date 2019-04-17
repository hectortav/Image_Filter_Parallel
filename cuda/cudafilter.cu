#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cudafunctions.h"

int main (int argc, char **argv)
{
  const int rows=2520, columns=1920;
  FILE *my_image = NULL;
  int *d_pixel = NULL;
  int *d_receive = NULL;
  int *h_pixel = NULL;
  float filter[3][3] = {{0.0625, 0.125, 0.0625},
                     {0.125, 0.25, 0.125},
                     {0.0625, 0.125, 0.0625}};
  float *d_filter, *h_filter;
  int i, j;
  float norm;
  int threadX, threadY, threadZ;
  float time;
  cudaEvent_t start, stop;

  //https://devtalk.nvidia.com/default/topic/978550/cuda-programming-and-performance/maximum-number-of-threads-on-thread-block/

  if (atoi(argv[1]) < 1)
  {
    printf("Worng arguments\n");
    return -1;
  }

  threadX = atoi(argv[1]);

  if (atoi(argv[2]) < 1)
  {
    printf("Worng arguments\n");
    return -1;
  }
  threadY = atoi(argv[2]);

  if (atoi(argv[3]) < 1 || atoi(argv[3]) > 64)
  {
    printf("Worng arguments\n");
    return -1;
  }

  threadZ = atoi(argv[3]);

  if (threadX * threadY *threadZ > 1024)
  {
    printf("Too many threads\n");
    return -1;
  }

  if (argc != 5)
  {
    printf("Not enough arguments\n");
    return -1;
  }
  if (strcmp(argv[4], "grayscale") == 0)
  {
    my_image = fopen("../waterfall_grey_1920_2520.raw", "rb");
    if (my_image == NULL)
    {
      perror("Could not open input file");
      return -1;
    }

    h_filter = (float *)malloc((3 * 3) * sizeof( float));
    h_pixel = (int *)malloc((rows * columns) * sizeof( int));
    cudaMalloc((void **)&d_pixel, (rows * columns) * sizeof( int));
    cudaMalloc((void **)&d_receive, (rows * columns) * sizeof( int));
    cudaMalloc((void **)&d_filter, (3 * 3) * sizeof( float));



    for (i = 0; i < (rows * columns); i++)
    {
      fread(&h_pixel[i], 1, 1, my_image);
    }
    fclose(my_image);

    cudaMemcpy(d_pixel, h_pixel, rows * columns * sizeof(int), cudaMemcpyHostToDevice);


    norm = 0.0;
    for(i = 0; i < 3; i++)
    {
      for(j = 0; j < 3; j++)
      {
        h_filter[i*3 + j] = filter[i][j];
        norm += h_filter[i*3 + j];
      }
    }

    cudaMemcpy(d_filter, h_filter, 3 * 3 * sizeof(float), cudaMemcpyHostToDevice);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    dim3 blockDim(threadX, threadY, threadZ);
    dim3 gridDim((rows + blockDim.x - 1)/ blockDim.x, (columns + blockDim.y - 1) / blockDim.y, 1);
    grayscale_cuda<<< gridDim, blockDim, 0 >>>(rows, columns, d_pixel, d_filter, norm, d_receive);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    cudaMemcpy(h_pixel, d_receive, rows * columns * sizeof(int), cudaMemcpyDeviceToHost);

    my_image = fopen("cuda_image_grayscale.raw","wb");
    if (my_image == NULL)
    {
      perror("Could not open input file");
      return -1;
    }
    for (i = 0; i < (rows * columns) - 1; i++)
    {
      fwrite(&h_pixel[i], 1, 1, my_image);
    }
    fclose(my_image);

    //printf("Time to generate:  %3.1f ms\n", time);
    printf("%3.1f, %d, %d, %d\n",time, threadX, threadY, threadZ);

    free(my_image);
    cudaFree(d_pixel);
    cudaFree(d_receive);
    cudaFree(d_filter);
    free(h_pixel);
    free(h_filter);
    return 0;
  }
  else if (strcmp(argv[4], "rgb") == 0)
  {
    my_image = fopen("../waterfall_1920_2520.raw", "rb");
    if (my_image == NULL){
      perror("Could not open input file");
      return -1;
    }

    h_filter = (float *)malloc((3 * 3) * sizeof( float));
    h_pixel = (int *)malloc((rows * columns * 3) * sizeof( int));
    cudaMalloc((void **)&d_pixel, (rows * columns * 3) * sizeof( int));
    cudaMalloc((void **)&d_receive, (rows * columns * 3) * sizeof( int));
    cudaMalloc((void **)&d_filter, (3 * 3) * sizeof( float));

    for (i = 0; i < (rows * columns * 3) - 1; i++)
    {
      fread(&h_pixel[i], 1, 1, my_image);
    }
    fclose(my_image);

    cudaMemcpy(d_pixel, h_pixel, rows * columns * 3 * sizeof(int), cudaMemcpyHostToDevice);

    norm = 0.0;
    for(i = 0; i < 3; i++)
    {
      for(j = 0; j < 3; j++)
      {
        h_filter[i*3 + j] = filter[i][j];
        norm += h_filter[i*3 + j];
      }
    }

    cudaMemcpy(d_filter, h_filter, 3 * 3 * sizeof(float), cudaMemcpyHostToDevice);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    dim3 blockDim(threadX, threadY, threadZ);
    dim3 gridDim((rows + blockDim.x - 1)/ blockDim.x, (columns * 3 + blockDim.y - 1) / blockDim.y, 1);
    rgb_cuda<<< gridDim, blockDim, 0 >>>(rows, columns, d_pixel, d_filter, norm, d_receive); //stelnw sketo columns
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    cudaMemcpy(h_pixel, d_receive, rows * columns * 3 * sizeof(int), cudaMemcpyDeviceToHost);

    my_image = fopen("cuda_image_rgb.raw","wb");
    if (my_image == NULL)
    {
      perror("Could not open input file");
      return -1;
    }
    for (i = 0; i < (rows * columns * 3); i++)
    {
      fwrite(&h_pixel[i], 1, 1, my_image);
    }
    fclose(my_image);

    printf("Time to generate:  %3.1f ms\n", time);
    //printf("%3.1f, %d, %d, %d\n",time, threadX, threadY, threadZ);

    free(my_image);
    cudaFree(d_pixel);
    cudaFree(d_receive);
    cudaFree(d_filter);
    free(h_pixel);
    free(h_filter);
    return 0;
  }
  else
  {
    printf("Worng arguments\n");
    return -1;
  }
}
