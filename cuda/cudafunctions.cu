#include "cudafunctions.h"
#include <stdio.h>

__global__ void grayscale_cuda(int rows, int columns,  int* pixel, float* filter,  float norm, int* receive)
{
  int i, j, x, y;
  int cube[3][3];
  float sum;

  i= blockIdx.x * blockDim.x + threadIdx.x;
  j= blockIdx.y * blockDim.y + threadIdx.y;

  if (i >= rows || j >= columns)
    return ;

    cube[0][0] = pixel[((i - 1) * columns) + j - 1]; //ul
    cube[1][0] = pixel[(i * columns) + j - 1]; //l= {{-2, -2, -2},
    cube[2][0] = pixel[((i + 1) * columns) + j - 1]; //dl
    cube[0][2] = pixel[((i - 1) * columns) + j + 1]; //ur
    cube[1][2] = pixel[(i * columns) + j + 1]; //r
    cube[2][2] = pixel[((i + 1) * columns) + j + 1]; //dr
    cube[0][1] = pixel[((i - 1) * columns) + j]; //u
    cube[1][1] = pixel[(i * columns) + j]; //c
    cube[2][1] = pixel[((i + 1) * columns) + j]; //d

      if (j == 0)
      {
        cube[0][0] = -1;
        cube[1][0] = -1;
        cube[2][0] = -1;

      }
      if (i == 0)
      {
        cube[0][0] = -1;
        cube[0][1] = -1;
        cube[0][2] = -1;
      }
      if (i == rows - 1)
      {
        cube[2][0] = -1;
        cube[2][1] = -1;
        cube[2][2] = -1;
      }
      if (j == (columns - 1))
      {
        cube[0][2] = -1;
        cube[1][2] = -1;
        cube[2][2] = -1;
      }

      sum = 0.0;
      for(x = 0; x < 3; x++)
      {
        for(y = 0; y < 3; y++)
        {
          if (cube[x][y] == -1)
          {
            cube[x][y] = cube[1][1];
          }
          sum += (cube[x][y] * filter[x*3 + y] / norm);
          if (cube[x][y] > 255 || cube[x][y] < 0)
            printf("cube[%d][%d] = %d * filter[%d*3 + %d] = %f / norm = %f\n", x, y, cube[x][y], x, y, filter[x*3 + y], norm);

        }
      }

      receive[i*columns + j] = (int)sum;

}

__global__ void rgb_cuda(int rows, int columns,  int* pixel, float* filter,  float norm, int* receive)
{
  int i, j, x, y;
  int cube[3][3];
  float sum, colour;

  i= blockIdx.x * blockDim.x + threadIdx.x;
  j= blockIdx.y * blockDim.y + threadIdx.y;

  if (i >= rows || j >= columns * 3)
    return ;

    colour = j % 3;
    cube[0][0] = pixel[((i - 1) * columns * 3) + j - 3]; //ul
    cube[1][0] = pixel[(i * columns * 3) + j - 3]; //l= {{-2, -2, -2},
    cube[2][0] = pixel[((i + 1) * columns * 3) + j - 3]; //dl
    cube[0][2] = pixel[((i - 1) * columns * 3) + j + 3]; //ur
    cube[1][2] = pixel[(i * columns * 3) + j + 3]; //r
    cube[2][2] = pixel[((i + 1) * columns * 3) + j + 3]; //dr
    cube[0][1] = pixel[((i - 1) * columns * 3) + j]; //u
    cube[1][1] = pixel[(i * columns * 3) + j]; //c
    cube[2][1] = pixel[((i + 1) * columns * 3) + j]; //d

      if (j - colour== 0)
      {
        cube[0][0] = -1;
        cube[1][0] = -1;
        cube[2][0] = -1;

      }
      if (i == 0)
      {
        cube[0][0] = -1;
        cube[0][1] = -1;
        cube[0][2] = -1;
      }
      if (i == rows - 1)
      {
        cube[2][0] = -1;
        cube[2][1] = -1;
        cube[2][2] = -1;
      }
      if (j - colour == (columns * 3 - 1))
      {
        cube[0][2] = -1;
        cube[1][2] = -1;
        cube[2][2] = -1;
      }

      sum = 0.0;
      for(x = 0; x < 3; x++)
      {
        for(y = 0; y < 3; y++)
        {
          if (cube[x][y] == -1)
          {
            cube[x][y] = cube[1][1];
          }
          sum += (cube[x][y] * filter[x*3 + y] / norm);
          if (cube[x][y] > 255 || cube[x][y] < 0)
            printf("cube[%d][%d] = %d * filter[%d*3 + %d] = %f / norm = %f\n", x, y, cube[x][y], x, y, filter[x*3 + y], norm);

        }
      }
      
      receive[i*columns*3 + j] = (int)sum;

}
