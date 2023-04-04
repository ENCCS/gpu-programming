#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "heat.h"

/* Initialize the heat equation solver */
void initialize(int argc, char *argv[], field *current,
                field *previous, int *nsteps)
{
    int rows = ROWS;
    int cols = COLS;
    *nsteps = NSTEPS;

    switch (argc) {
    case 1:
        /* Use default values */
        break;
    case 4:
        /* Field dimensions */
        rows = atoi(argv[1]);
        cols = atoi(argv[2]);
        /* Number of time steps */
        *nsteps = atoi(argv[3]);
        break;
    default:
        printf("Unsupported number of command line arguments\n");
        exit(-1);
    }

	field_create(current, previous, rows, cols);
}

void field_create (field *heat1, field *heat2, int rows, int cols)
{
	field_set_size(heat1, rows, cols);
	field_generate(heat1);

	field_set_size(heat2, rows, cols);
	field_allocate(heat2);

	field_copy(heat1, heat2);
}

/* Generate initial temperature field.  Pattern is disc with a radius
 * of nx / 6 in the center of the grid.
 * Boundary conditions are (different) constant temperatures outside the grid */
void field_generate(field *heat)
{
    int ind;
    double radius;
    int dx, dy;

    /* Allocate the temperature array */
    field_allocate(heat);

    /* Radius of the source disc */
    radius = heat->nx / 6.0;
    for (int i = 0; i < heat->nx + 2; i++) {
        for (int j = 0; j < heat->ny + 2; j++) {
	    ind = i * (heat->ny + 2) + j;
            /* Distance of point i, j from the origin */
            dx = i - heat->nx / 2 + 1;
            dy = j - heat->ny / 2 + 1;
            if (dx * dx + dy * dy < radius * radius) {
                heat->data[ind] = T_DISC;
            } else {
                heat->data[ind] = T_AREA;
            }
        }
    }

    /* Boundary conditions */
    for (int i = 0; i < heat->nx + 2; i++) {
        heat->data[i * (heat->ny + 2)] = T_LEFT;
        heat->data[i * (heat->ny + 2) + heat->ny + 1] = T_RIGHT;
    }

    for (int j = 0; j < heat->ny + 2; j++) {
        heat->data[j] = T_UPPER;
    }
    for (int j = 0; j < heat->ny + 2; j++) {
        heat->data[(heat->nx + 1) * (heat->ny + 2) + j] = T_LOWER;
    }
}

/* Set dimensions of the field. Note that the nx is the size of the first
 * dimension and ny the second. */
void field_set_size(field *heat, int nx, int ny)
{
    heat->dx = DX;
    heat->dy = DY;
    heat->nx = nx;
    heat->ny = ny;
}
