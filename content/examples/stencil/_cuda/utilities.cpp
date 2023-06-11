// Utility functions for heat equation solver

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <mpi.h>

#include "heat.h"


// Copy data on heat1 into heat2
void field_copy(field *heat1, field *heat2)
{
    assert(heat1->nx == heat2->nx);
    assert(heat1->ny == heat2->ny);
    assert(heat1->data.size() == heat2->data.size());
    std::copy(heat1->data.begin(), heat1->data.end(), heat2->data.begin());
}


// Swap the field data for heat1 and heat2
void field_swap(field *heat1, field *heat2)
{
    assert(heat1->nx == heat2->nx);
    assert(heat1->ny == heat2->ny);
    assert(heat1->data.size() == heat2->data.size());
    std::swap(heat1->data, heat2->data);
}


// Allocate memory for a temperature field and initialise it to zero
void field_allocate(field *heat)
{
    // Allocate also ghost layers
    heat->data = new double [(heat->nx + 2) * (heat->ny + 2)];

    // Initialize to zero
    memset(heat->data, 0.0, (heat->nx + 2) * (heat->ny + 2) * sizeof(double));
}


// Calculate average temperature over the non-boundary grid cells
double average(field *heat)
{
    double local_average = 0.0;
    double average = 0.0;

    for (int i = 1; i < heat->nx + 1; i++)
    {
        for (int j = 1; j < heat->ny + 1; j++)
        {
            int ind = i * (heat->ny + 2) + j;
            local_average += heat->data[ind];
        }
    }

    MPI_Allreduce(&local_average, &average, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    average /= (heat->nx_full * heat->ny_full);
    return average;
}

