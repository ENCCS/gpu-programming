// (c) 2023 ENCCS, CSC and the contributors
#include "heat.h"
#include <sycl/sycl.hpp>

// Update the temperature values using five-point stencil
// Arguments:
//   queue: SYCL queue
//   curr: current temperature values
//   prev: temperature values from previous time step
//   a: diffusivity
//   dt: time step
void evolve(sycl::queue &Q, field *curr, field *prev, double a, double dt) {
  // Help the compiler avoid being confused by the structs
  int nx = prev->nx;
  int ny = prev->ny;
  int size = (nx + 2) * (ny + 2);

  // Determine the temperature field at next time step
  // As we have fixed boundary conditions, the outermost gridpoints
  // are not updated.
  double dx2 = prev->dx * prev->dx;
  double dy2 = prev->dy * prev->dy;

  double *currdata = sycl::malloc_device<double>(size, Q);
  double *prevdata = sycl::malloc_device<double>(size, Q);
  Q.copy<double>(curr->data.data(), currdata, size);
  Q.copy<double>(prev->data.data(), prevdata, size);

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

  Q.copy<double>(currdata, curr->data.data(), size).wait();
  sycl::free(currdata, Q);
  sycl::free(prevdata, Q);
}
