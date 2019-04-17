#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mpifunctions.h"

int main (int argc, char **argv)
{
  const int rows=2520, columns=1920;

  int *pixel = NULL;
  int *quavo = NULL;
  int *receive = NULL;
  int *tmp = NULL;
  int cube[3][3] = {0};
  float filter[3][3] = {{0.0625, 0.125, 0.0625},
                      {0.125, 0.25, 0.125},
                      {0.0625, 0.125, 0.0625}};

  FILE *my_image = NULL;
  FILE *new_image = NULL;

  int rank, size, i, j, id, tasks, workers, drows, offset, dest, source, x, y, colour, loops, l, remains;
  float norm, sum;
  double end, start;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &id);
  MPI_Comm_size(MPI_COMM_WORLD, &tasks);

  start = MPI_Wtime();

  if (argc != 4)
  {
    printf("Not enough arguments\n");
    MPI_Finalize();
    return -1;
  }
  if ((strcmp(argv[1], "parallel") != 0) && strcmp(argv[1], "serial") != 0)
  {
    printf("Wrong arguments\n");
    MPI_Finalize();
    return -1;
  }
  if ((strcmp(argv[1], "parallel") == 0) && (tasks < 3))
  {
    printf("We need at least 3 tasks to run in parallel\n");
    MPI_Finalize();
    return -1;
  }
  if ((strcmp(argv[2], "grayscale") != 0) && (strcmp(argv[2], "rgb") != 0))
  {
    printf("Worng arguments\n");
    MPI_Finalize();
    return -1;
  }
  if (atoi(argv[3]) < 1)
  {
    printf("Worng arguments\n");
    MPI_Finalize();
    return -1;
  }

  loops = atoi(argv[3]);

  workers = tasks - 1;

  if ((strcmp(argv[1], "parallel") == 0) && (strcmp(argv[2], "grayscale") == 0))
  {
    drows = rows / workers;
    remains = rows % workers;
    pixel = (int *)malloc((rows * columns) * sizeof( int));
    receive = (int *)malloc(((drows + remains + 2) * columns) * sizeof(int));

    if (id == 0)
    {
      my_image = fopen("../waterfall_grey_1920_2520.raw", "rb");
      if (my_image == NULL){
        perror("Could not open input file");
        return -1;
      }
      for (i = 0; i < (rows * columns); i++)
      {
        fread(&pixel[i], 1, 1, my_image);
      }
      fclose(my_image);
    }

    for (l = 0; l < loops; l++)
    {
      if (id == 0)
      {
        offset = 0;
        for(dest = 1; dest < tasks; dest++)
        {
          if (dest == tasks - 1)
          {
            drows += remains;
          }
          MPI_Send(&offset, 1, MPI_INT, dest, 1, MPI_COMM_WORLD);
          MPI_Send(&pixel[offset], (drows * columns), MPI_INT, dest, 2, MPI_COMM_WORLD);
          if (dest == 1)
          {
            MPI_Send(&pixel[offset + (drows * columns)], columns, MPI_INT, dest, 4, MPI_COMM_WORLD);
          }
          else if (dest == tasks - 1)
          {
            MPI_Send(&pixel[offset - columns], columns, MPI_INT, dest, 3, MPI_COMM_WORLD);
          }
          else
          {
            MPI_Send(&pixel[offset - columns], columns, MPI_INT, dest, 3, MPI_COMM_WORLD);
            MPI_Send(&pixel[offset + (drows * columns)], columns, MPI_INT, dest, 4, MPI_COMM_WORLD);
          }
          offset += drows * columns;
        }

        for(source = 1; source < tasks; source++)
        {
          if (source == tasks - 1)
          {
            drows += remains;
          }
          MPI_Recv(&offset, 1, MPI_INT, source, 5, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          MPI_Recv(&pixel[offset], (drows) * (columns), MPI_INT, source, 6, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        new_image = fopen("parallel_grayscale_image.raw","wb");
        for (i = 0; i < (rows * columns); i++)
        {
          fwrite(&pixel[i], 1, 1, new_image);
        }
        fclose(new_image);
      }

      if (id > 0)
      {
        if (id == tasks - 1)
        {
          drows += remains;
        }
        source = 0;
        MPI_Recv(&offset, 1, MPI_INT, source, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&receive[columns], (drows * columns), MPI_INT, source, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        if (offset == 0)
        {
          MPI_Recv(&receive[drows*columns + columns], columns, MPI_INT, source, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          for (i = 0; i < columns; i++)
            receive[i] = -1;
        }
        else if (offset == columns * rows - drows * columns)
        {
          MPI_Recv(&receive[0], columns, MPI_INT, source, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          for (i = 0; i < columns; i++)
            receive[i + drows*columns + columns] = -1;
        }
        else
        {
          MPI_Recv(&receive[0], columns, MPI_INT, source, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          MPI_Recv(&receive[drows*columns + columns], columns, MPI_INT, source, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        grayscale_parallel(drows, columns, receive, filter, pixel);

        MPI_Send(&offset, 1, MPI_INT, 0, 5, MPI_COMM_WORLD);
        MPI_Send(&pixel[0], (drows * columns), MPI_INT, 0, 6, MPI_COMM_WORLD);
      }
    }

    free(receive);
    free(pixel);
  } else if (((strcmp(argv[1], "parallel") == 0) && (strcmp(argv[2], "rgb") == 0)))
  {
    drows = rows / workers;
    remains = rows % workers;
    pixel = (int *)malloc((rows * columns * 3) * sizeof( int));
    receive = (int *)malloc(((drows + 2 + remains) * columns * 3) * sizeof( int));

    if (id == 0)
    {
      my_image = fopen("../waterfall_1920_2520.raw", "rb");
      if (my_image == NULL){
        perror("Could not open input file");
        return -1;
      }
      for (i = 0; i < (rows * columns * 3); i++)
      {
        fread(&pixel[i], 1, 1, my_image);
      }
      fclose(my_image);
    }

    for (l = 0; l < loops; l++)
    {
      if (id == 0)
      {
        if (dest == tasks - 1)
        {
          drows += remains;
        }
        offset = 0;
        for(dest = 1; dest < tasks; dest++)
        {
          MPI_Send(&offset, 1, MPI_INT, dest, 1, MPI_COMM_WORLD);
          MPI_Send(&pixel[offset], (drows * columns * 3), MPI_INT, dest, 2, MPI_COMM_WORLD);
          if (dest == 1)
          {
            MPI_Send(&pixel[offset + (drows * columns * 3)], (columns * 3), MPI_INT, dest, 4, MPI_COMM_WORLD);
          }
          else if (dest == tasks - 1)
          {
            MPI_Send(&pixel[offset - (columns * 3)], (columns * 3), MPI_INT, dest, 3, MPI_COMM_WORLD);
          }
          else
          {
            MPI_Send(&pixel[offset - (columns * 3)], (columns * 3), MPI_INT, dest, 3, MPI_COMM_WORLD);
            MPI_Send(&pixel[offset + (drows * columns * 3)], (columns * 3), MPI_INT, dest, 4, MPI_COMM_WORLD);
          }
          offset += drows * columns * 3;
        }

        for(source = 1; source < tasks; source++)
        {
          if (source == tasks - 1)
          {
            drows += remains;
          }
          MPI_Recv(&offset, 1, MPI_INT, source, 5, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          MPI_Recv(&pixel[offset], (drows * columns * 3), MPI_INT, source, 6, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        new_image = fopen("parallel_rgb_image.raw","wb");
        for (i = 0; i < (rows * columns * 3); i++)
        {
          fwrite(&pixel[i], 1, 1, new_image);
        }
        fclose(new_image);
      }

      if (id > 0)
      {
        if (id == tasks - 1)
        {
          drows += remains;
        }
        source = 0;
        MPI_Recv(&offset, 1, MPI_INT, source, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&receive[columns * 3], (drows * columns * 3), MPI_INT, source, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        if (offset == 0)
        {
          MPI_Recv(&receive[drows*columns*3 + columns*3], columns * 3, MPI_INT, source, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          for (i = 0; i < columns * 3; i++)
            receive[i] = -1;
        }
        else if (offset == columns * 3 * rows - drows * columns * 3)
        {
          MPI_Recv(&receive[0], columns * 3, MPI_INT, source, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          for (i = 0; i < columns * 3; i++)
            receive[i + drows*columns*3 + columns*3] = -1;
        }
        else
        {
          MPI_Recv(&receive[0], columns * 3, MPI_INT, source, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          MPI_Recv(&receive[drows*columns*3 + columns*3], columns*3, MPI_INT, source, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        rgb_parallel(drows, columns, receive, filter, pixel);

        MPI_Send(&offset, 1, MPI_INT, 0, 5, MPI_COMM_WORLD);
        MPI_Send(&pixel[0], (drows * columns * 3), MPI_INT, 0, 6, MPI_COMM_WORLD);
      }
    }

    free(pixel);
    free(receive);
  } else if (((strcmp(argv[1], "serial") == 0) && (strcmp(argv[2], "grayscale") == 0)))
  {
    if (id == 0)
    {
      pixel = ( int *)malloc((rows * columns) * sizeof( int));

      my_image = fopen("../waterfall_grey_1920_2520.raw", "rb");
      if (my_image == NULL){
        perror("Could not open input file");
        return -1;
      }
      for (i = 0; i < (rows * columns); i++)
      {
        fread(&pixel[i], 1, 1, my_image);
      }
      fclose(my_image);

      for (l = 0; l < loops; l++)
      {
        quavo = ( int *)malloc((rows * columns) * sizeof( int));

        grayscale_serial(rows, columns, quavo, filter, pixel);

        new_image = fopen("serial_grayscale_image.raw","wb");
        if (new_image == NULL){
          perror("Could not open input file");
          return -1;
        }
        for (i = 0; i < (rows * columns); i++)
        {
          fwrite(&quavo[i], 1, 1, new_image);
          pixel[i] = quavo[i];
        }
        fclose(new_image);

        free(quavo);
      }
      free(pixel);
    }
  } else if (((strcmp(argv[1], "serial") == 0) && (strcmp(argv[2], "rgb") == 0)))
  {
    if (id == 0)
    {
      pixel = (int *)malloc((rows * columns * 3) * sizeof(int));

      my_image = fopen("../waterfall_1920_2520.raw", "rb");
      if (my_image == NULL){
        perror("Could not open input file");
        return -1;
      }
      for (i = 0; i < (rows * columns * 3); i++)
      {
        fread(&pixel[i], 1, 1, my_image);
      }
      fclose(my_image);

      for (l = 0; l < loops; l++)
      {
        quavo = (int *)malloc((rows * columns * 3) * sizeof(int));

        rgb_serial(rows, columns, quavo, filter, pixel);

        new_image = fopen("serial_rgb.raw","wb");
        if (new_image == NULL){
          perror("Could not open input file");
          return -1;
        }
        for (i = 0; i < (rows * columns * 3); i++)
        {
          fwrite(&quavo[i], 1, 1, new_image);
          pixel[i] = quavo[i];
        }
        fclose(new_image);

        free(quavo);
      }
      free(pixel);
    }
  }

  end = MPI_Wtime();

  MPI_Finalize();

  printf("%f\n", (end - start));

  return 0;
}
