#include "heat.h"
#include <sycl/sycl.hpp>
using namespace sycl;

// Update the temperature values using five-point stencil
// Arguments:
//   queue: SYCL queue
//   curr: current temperature values
//   prev: temperature values from previous time step
//   a: diffusivity
//   dt: time step
void evolve(queue &Q, field *curr, field *prev, double a, double dt)
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
    buffer<double, 2> buf_curr { currdata, range<2>(nx + 2, ny + 2) },
                      buf_prev { prevdata, range<2>(nx + 2, ny + 2) };

    Q.submit([&](handler &cgh) {
      auto acc_curr = accessor(buf_curr, cgh, read_write);
      auto acc_prev = accessor(buf_prev, cgh, read_only);

      cgh.parallel_for(range<2>(nx, ny), [=](id<2> id) {
        auto j = id[0] + 1;
        auto i = id[1] + 1;
        acc_curr[j][i] = acc_prev[j][i] + a * dt *
            ((acc_prev[j][i + 1] - 2.0 * acc_prev[j][i] + acc_prev[j][i - 1]) / dx2 +
             (acc_prev[j + 1][i] - 2.0 * acc_prev[j][i] + acc_prev[j - 1][i]) / dy2);
      });
    });
  }
}
