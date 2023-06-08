// Main routine for heat equation solver in 2D.
// (c) 2023 ENCCS, CSC and the contributors
#include <cstdio>
#include <omp.h>

#include "heat.h"

double start_time () { return omp_get_wtime(); }
double stop_time () { return omp_get_wtime(); }


int main(int argc, char **argv)
{
    // Set up the solver
    int nsteps;
    field current, previous;
    initialize(argc, argv, &current, &previous, &nsteps);

    // Output the initial field and its temperature
    field_write(&current, 0);
    double average_temp = field_average(&current);
    printf("Average temperature, start: %f\n", average_temp);

    // Set diffusivity constant
    double a = 0.5;
    // Compute the largest stable time step
    double dx2 = current.dx * current.dx;
    double dy2 = current.dy * current.dy;
    double dt = dx2 * dy2 / (2.0 * a * (dx2 + dy2));
    // Set output interval
    int output_interval = 1500;

    // Start timer
    double start_clock = start_time();
    // Time evolution
    for (int iter = 1; iter <= nsteps; iter++) {
        evolve(&current, &previous, a, dt);
        if (iter % output_interval == 0) {
            field_write(&current, iter);
        }
        // Swap current and previous fields for next iteration step
        field_swap(&current, &previous);
    }
    // Stop timer
    double stop_clock = stop_time();

    // Output the final field and its temperature
    average_temp = field_average(&previous);
    printf("Average temperature at end: %f\n", average_temp);
    // Compare temperature for reference
    if (argc == 1) {
        printf("Control temperature at end: 59.281239\n");
    }
    field_write(&previous, nsteps);

    // Determine the computation time used for all the iterations
    printf("Iterations took %.3f seconds.\n", (stop_clock - start_clock));

    return 0;
}
