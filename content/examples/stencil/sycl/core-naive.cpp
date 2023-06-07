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
void evolve(sycl::queue &Q, 
            field *curr, field *prev, double a, double dt)
{
  // Help the compiler avoid being confused by the structs
  double *currdata = curr->data.data();
  double *prevdata = prev->data.data();
  int nx = prev->nx;
  int ny = prev->ny;

  // Determine the temperature field at next time step
  // As we have fixed boundary conditions, the outermost gridpoints
  // are not updated.
  double dx2 = prev->dx * prev->dx;
  double dy2 = prev->dy * prev->dy;

  {
    sycl::buffer<double, 2> buf_curr { currdata, sycl::range<2>(nx + 2, ny + 2) },
                            buf_prev { prevdata, sycl::range<2>(nx + 2, ny + 2) };

    Q.submit([&](sycl::handler &cgh) {
      auto acc_curr = sycl::accessor(buf_curr, cgh, sycl::read_write);
      auto acc_prev = sycl::accessor(buf_prev, cgh, sycl::read_only);

      cgh.parallel_for(sycl::range<2>(nx, ny), [=](sycl::id<2> id) {
        auto j = id[0] + 1;
        auto i = id[1] + 1;
        acc_curr[j][i] = acc_prev[j][i] + a * dt *
            ((acc_prev[j][i + 1] - 2.0 * acc_prev[j][i] + acc_prev[j][i - 1]) / dx2 +
             (acc_prev[j + 1][i] - 2.0 * acc_prev[j][i] + acc_prev[j - 1][i]) / dy2);
      });
    });
  }
  // Data is automatically copied back to the CPU when buffers go out of scope
}
