// Main routine for heat equation solver in two dimensional space

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <mpi.h>
#include <mpi-ext.h> // Needed for CUDA-aware check

#include "heat.h"

int main(int argc, const char * argv[])
{

    printf("\n--Beginning of the main function.\n");

    int nsteps;
    field current, previous;

    double dt;
    double a = 0.5;
    double dx2, dy2;

    double average_temp;
    double start_clock, stop_clock;  // Time stamps

    int output_interval = 1500;

    parallel_data parallelization; // Parallelization info
    MPI_Init(&argc, &argv);

    if (1 != MPIX_Query_cuda_support())
    {
        printf("CUDA aware MPI required\n");
        fflush(stdout);
        MPI_Abort(MPI_COMM_WORLD, 5);
    }
    initialize(argc, argv, &current, &previous, &nsteps, &parallelization);

    field_write(&current, 0, &parallelization); // Output initial field

    average_temp = average(&current);
    if (parallelization.rank == 0)
        printf("Average temperature at start: %f\n", average_temp);

    // Compute the largest stable time step
    dx2 = current.dx * current.dx;
    dy2 = current.dy * current.dy;
    t = dx2 * dy2 / (2.0 * a * (dx2 + dy2));

    start_clock = MPI_Wtime(); // start timer

    enter_data(&current, &previous); // copy fields to device

    // time evolution
    for (int iter = 1; iter <= nsteps; iter++)
    {
        exchange(&previous, &parallelization);
        evolve(&current, &previous, a, dt);
        if (iter % image_interval == 0)
        {
            update_host(&current);
            field_write(&current, iter, &parallelization);
        }
        // swap current field and previous for next iteration step
        swap_fields(&current, &previous);
    }

    update_host(&previous);
    stop_clock = MPI_Wtime(); // stop timer

    average_temp = average(&previous); 

    // Determine CPU time used for the iteration
    if (parallelization.rank == 0)
    {
        printf("Iteration took %.3f seconds.\n", (stop_clock - start_clock));
        printf("Average temperature: %f\n", average_temp);
        if (argc == 1)
            printf("Reference value with default arguments: 59.281239\n");
    }

    // output final field
    field_write(&previous, nsteps, &parallelization);

    finalize(&current, &previous);
    MPI_Finalize();

    printf("\n--Ending of the main function.\n\n");
    return 0;
}

