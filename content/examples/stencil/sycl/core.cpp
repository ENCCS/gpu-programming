// (c) 2023 ENCCS, CSC and the contributors
#include "heat.h"
#include <sycl/sycl.hpp>

// Update the temperature values using five-point stencil
// Arguments:
//   queue: SYCL queue
//   currdata: current temperature values (device pointer)
//   prevdata: temperature values from previous time step (device pointer)
//   prev: description of the grid parameters
//   a: diffusivity
//   dt: time step
void evolve(sycl::queue &Q, double* currdata, const double* prevdata,
            const field *prev, double a, double dt)
{
  int nx = prev->nx;
  int ny = prev->ny;
  
  // Determine the temperature field at next time step
  // As we have fixed boundary conditions, the outermost gridpoints
  // are not updated.
  double dx2 = prev->dx * prev->dx;
  double dy2 = prev->dy * prev->dy;

  Q.parallel_for(sycl::range<2>(nx, ny), [=](sycl::id<2> id) {
    auto i = id[0] + 1;
    auto j = id[1] + 1;

    int ind = i * (ny + 2) + j;
    int ip = (i + 1) * (ny + 2) + j;
    int im = (i - 1) * (ny + 2) + j;
    int jp = i * (ny + 2) + j + 1;
    int jm = i * (ny + 2) + j - 1;
    currdata[ind] = prevdata[ind] + a*dt*
      ((prevdata[ip] - 2.0*prevdata[ind] + prevdata[im]) / dx2 +
       (prevdata[jp] - 2.0*prevdata[ind] + prevdata[jm]) / dy2);
  });
}

void copy_to_buffer(sycl::queue Q, double* buffer, const field* f)
{
    int size = (f->nx + 2) * (f->ny + 2);
    Q.copy<double>(f->data.data(), buffer, size);
}

void copy_from_buffer(sycl::queue Q, const double* buffer, field *f)
{
    int size = (f->nx + 2) * (f->ny + 2);
    Q.copy<double>(buffer, f->data.data(), size).wait();
}
