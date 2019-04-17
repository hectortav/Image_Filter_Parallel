#include "openmpfunctions.h"
#include <stdio.h>
#include "omp.h"

void grayscale_parallel(int drows, int columns,  int* receive, float filter[3][3],  int* pixel)
{
  int i, j, cube[3][3], x, y;
  float sum, norm;

  norm = 0.0;
  for(x = 0; x < 3; x++)
  {
    for(y = 0; y < 3; y++)
    {
      norm += filter[x][y];
    }
  }

  #pragma omp parallel for shared(drows, columns, receive, norm, filter, pixel) private(i, j, x, y, sum, cube) schedule(static) collapse(2)
  for (i = 1; i < drows + 2; i++)
  {
    for (j = 0; j < columns; j++)
    {
      if (j == 0)
      {
        cube[0][0] = -1;
        cube[1][0] = -1;
        cube[2][0] = -1;
      }
      else
      {
        cube[0][0] = receive[((i - 1) * columns) + j - 1]; //ul
        cube[1][0] = receive[(i * columns) + j - 1]; //l
        cube[2][0] = receive[((i + 1) * columns) + j - 1]; //dl
      }
      if (j == (columns - 1))
      {
        cube[0][2] = -1;
        cube[1][2] = -1;
        cube[2][2] = -1;
      }
      else
      {
        cube[0][2] = receive[((i - 1) * columns) + j + 1]; //ur
        cube[1][2] = receive[(i * columns) + j + 1]; //r
        cube[2][2] = receive[((i + 1) * columns) + j + 1]; //dr
      }

      cube[0][1] = receive[((i - 1) * columns) + j]; //u
      cube[1][1] = receive[(i * columns) + j]; //c
      cube[2][1] = receive[((i + 1) * columns) + j]; //d

      sum = 0.0;

      for(x = 0; x < 3; x++)
      {
        for(y = 0; y < 3; y++)
        {
          if (cube[x][y] == -1)
          {
            cube[x][y] = cube[1][1];
          }

          sum += (cube[x][y] * filter[x][y] / norm);
        }
      }

      pixel[(i - 1)*columns + j] = (int)sum;
    }
  }
}

void rgb_parallel(int drows, int columns,  int* receive, float filter[3][3],  int* pixel)
{
  int i, j, cube[3][3], x, y;
  float sum, norm, colour;

  norm = 0.0;
  for(x = 0; x < 3; x++)
  {
    for(y = 0; y < 3; y++)
    {
      norm += filter[x][y];
    }
  }

  #pragma omp parallel for shared(drows, columns, receive, norm, filter, pixel) private(i, j, x, y, sum, cube) schedule(static) collapse(2)
  for (i = 1; i < drows + 2; i++)
  {
    for (j = 0; j < columns * 3; j++)
    {
      colour = j % 3;
      if ((j - colour) == 0)
      {
        cube[0][0] = -1;
        cube[1][0] = -1;
        cube[2][0] = -1;
      }
      else
      {
        cube[0][0] = receive[((i - 1) * columns * 3) + j - 3]; //ul
        cube[1][0] = receive[(i * columns * 3) + j - 3]; //l
        cube[2][0] = receive[((i + 1) * columns * 3) + j - 3]; //dl
      }
      if (( j - colour) == ((columns * 3) - 1))
      {
        cube[0][2] = -1;
        cube[1][2] = -1;
        cube[2][2] = -1;
      }
      else
      {
        cube[0][2] = receive[((i - 1) * columns * 3) + j + 3]; //ur
        cube[1][2] = receive[(i * columns * 3) + j + 3]; //r
        cube[2][2] = receive[((i + 1) * columns * 3) + j + 3]; //dr
      }

      cube[0][1] = receive[((i - 1) * columns * 3) + j]; //u
      cube[1][1] = receive[(i * columns * 3) + j]; //c
      cube[2][1] = receive[((i + 1) * columns * 3) + j]; //d

      sum = 0.0;

      for(x = 0; x < 3; x++)
      {
        for(y = 0; y < 3; y++)
        {
          if (cube[x][y] == -1)
          {
            cube[x][y] = cube[1][1];
          }

          sum += (cube[x][y] * filter[x][y] / norm);
        }
      }

      pixel[(i - 1)*columns*3 + j] = (int)sum;
    }
  }
}

void grayscale_serial(int rows, int columns,  int* quavo, float filter[3][3],  int* pixel)
{
  int i, j, x, y;
  float sum, norm;
  int cube[3][3];

  norm = 0.0;
  for(x = 0; x < 3; x++)
  {
    for(y = 0; y < 3; y++)
    {
      norm += filter[x][y];
    }
  }

  for (i = 0; i < rows; i++)
  {
    for (j = 0; j < columns; j++)
    {
      for(x = 0; x < 3; x++)
      {
        for(y = 0; y < 3; y++)
        {
          cube[x][y] = -2;
        }
      }
      cube[1][1] = pixel[(i * columns) + j]; //c
      if (i == 0)
      {
        cube[0][0] = cube[1][1];
        cube[0][1] = cube[1][1];
        cube[0][2] = cube[1][1];
      }
      if (i == rows-1)
      {
        cube[2][0] = cube[1][1];
        cube[2][1] = cube[1][1];
        cube[2][2] = cube[1][1];
      }
      if (j == 0)
      {
        cube[0][0] = cube[1][1];
        cube[1][0] = cube[1][1];
        cube[2][0] = cube[1][1];
      }
      if (j == columns-1)
      {
        cube[0][2] = cube[1][1];
        cube[1][2] = cube[1][1];
        cube[2][2] = cube[1][1];
      }
      if (cube[2][1] == -2) {
        cube[2][1] = pixel[((i + 1) * columns) + j];
      }
      if (cube[0][1] == -2) {
        cube[0][1] = pixel[((i - 1) * columns) + j];
      }
      if (cube[1][0] == -2) {
        cube[1][0] = pixel[(i * columns) + j - 1];
      }
      if (cube[1][2] == -2) {
        cube[1][2] = pixel[(i * columns) + j + 1];
      }
      if (cube[0][0] == -2) {
        cube[0][0] = pixel[((i - 1) * columns) + j - 1];
      }
      if (cube[0][2] == -2) {
        cube[0][2] = pixel[((i - 1) * columns) + j + 1];
      }
      if (cube[2][0] == -2) {
        cube[2][0] = pixel[((i + 1) * columns) + j - 1];
      }
      if (cube[2][2] == -2) {
        cube[2][2] = pixel[((i + 1) * columns) + j + 1];
      }

      sum = 0.0;

      for(x = 0; x < 3; x++)
      {
        for(y = 0; y < 3; y++)
        {
          sum += (cube[x][y] * filter[x][y] / norm);
        }
      }

      quavo[(i*columns + j)] = (int)sum;
    }
  }
}

void rgb_serial(int rows, int columns,  int* quavo, float filter[3][3],  int* pixel)
{
  int i, j, cube[3][3], x, y;
  float sum, norm, colour;

  norm = 0.0;
  for(x = 0; x < 3; x++)
  {
    for(y = 0; y < 3; y++)
    {
      norm += filter[x][y];
    }
  }

  for (i = 0; i < rows; i++)
  {
    for (j = 0; j < columns * 3; j++)
    {
      colour = j % 3;

      for(x = 0; x < 3; x++)
      {
        for(y = 0; y < 3; y++)
        {
          cube[x][y] = -2;
        }
      }
      cube[1][1] = pixel[(i * columns * 3) + j]; //c
      if (i == 0)
      {
        cube[0][0] = cube[1][1];
        cube[0][1] = cube[1][1];
        cube[0][2] = cube[1][1];
      }
      if (i == rows-1)
      {
        cube[2][0] = cube[1][1];
        cube[2][1] = cube[1][1];
        cube[2][2] = cube[1][1];
      }
      if (j - colour == 0)
      {
        cube[0][0] = cube[1][1];
        cube[1][0] = cube[1][1];
        cube[2][0] = cube[1][1];
      }
      if (j - colour == columns * 3 - 1)
      {
        cube[0][2] = cube[1][1];
        cube[1][2] = cube[1][1];
        cube[2][2] = cube[1][1];
      }
      if (cube[2][1] == -2) {
        cube[2][1] = pixel[((i + 1) * columns * 3) + j];
      }
      if (cube[0][1] == -2) {
        cube[0][1] = pixel[((i - 1) * columns * 3) + j];
      }
      if (cube[1][0] == -2) {
        cube[1][0] = pixel[(i * columns * 3) + j - 3];
      }
      if (cube[1][2] == -2) {
        cube[1][2] = pixel[(i * columns * 3) + j + 3];
      }
      if (cube[0][0] == -2) {
        cube[0][0] = pixel[((i - 1) * columns * 3) + j - 3];
      }
      if (cube[0][2] == -2) {
        cube[0][2] = pixel[((i - 1) * columns * 3) + j + 3];
      }
      if (cube[2][0] == -2) {
        cube[2][0] = pixel[((i + 1) * columns * 3) + j - 3];
      }
      if (cube[2][2] == -2) {
        cube[2][2] = pixel[((i + 1) * columns * 3) + j + 3];
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

          sum += (cube[x][y] * filter[x][y] / norm);
        }
      }

      quavo[(i * columns * 3) + j] = (int)sum;
    }
  }
}
