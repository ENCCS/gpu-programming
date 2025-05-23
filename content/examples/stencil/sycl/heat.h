// (c) 2023 ENCCS, CSC and the contributors
#ifndef __HEAT_H__
#define __HEAT_H__

#include <sycl/sycl.hpp>
#include <vector>

// Datatype for temperature field
struct field {
    // nx and ny are the dimensions of the field. The array data
    // contains also ghost layers, so it will have dimensions nx+2 x ny+2
    int nx;
    int ny;
    // Size of the grid cells
    double dx;
    double dy;
    // The temperature values in the 2D grid
    std::vector<double> data;
};

// CONSTANTS
// Fixed grid spacing
const double DX = 0.01;
const double DY = 0.01;
// Default temperatures
const double T_DISC = 5.0;
const double T_AREA = 65.0;
const double T_UPPER = 85.0;
const double T_LOWER = 5.0;
const double T_LEFT = 20.0;
const double T_RIGHT = 70.0;
// Default problem size
const int ROWS = 2000;
const int COLS = 2000;
const int NSTEPS = 500;


// Function prototypes
void initialize(int argc, char *argv[], field *heat1,
                field *heat2, int *nsteps);

void evolve(sycl::queue &Q, field *curr, field *prev, double a, double dt);

void evolve(sycl::queue &Q, double* buf_curr, const double* buf_prev, 
            const field *prev, double a, double dt);

void field_set_size(field *heat, int nx, int ny);

void field_generate(field *heat);

double field_average(field *heat);

void field_write(field *heat, int iter);

void field_create(field *heat1, field *heat2, int rows, int cols);

void field_copy(field *heat1, field *heat2);

void field_swap(field *heat1, field *heat2);

void field_allocate(field *heat);

// Data movement function prototypes
void copy_to_buffer(sycl::queue Q, double* buffer, const field* f);

void copy_from_buffer(sycl::queue Q, const double* buffer, field *f);

#endif  // __HEAT_H__
