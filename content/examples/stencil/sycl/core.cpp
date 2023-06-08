// (c) 2023 ENCCS, CSC and the contributors
#include "heat.h"
#include <sycl/sycl.hpp>

// Update the temperature values using five-point stencil
// Arguments:
//   queue: SYCL queue
//   d_curr: current temperature values
//   d_prev: temperature values from previous time step
//   prev: description of the grid parameters
//   a: diffusivity
//   dt: time step
void evolve(sycl::queue &Q, sycl::buffer<double, 2> d_curr, sycl::buffer<double, 2> d_prev,
            const field *prev, double a, double dt)
{
  int nx = prev->nx;
  int ny = prev->ny;
  
  // Determine the temperature field at next time step
  // As we have fixed boundary conditions, the outermost gridpoints
  // are not updated.
  double dx2 = prev->dx * prev->dx;
  double dy2 = prev->dy * prev->dy;

  {
    Q.submit([&](sycl::handler &cgh) {
      auto acc_curr = sycl::accessor(d_curr, cgh, sycl::read_write);
      auto acc_prev = sycl::accessor(d_prev, cgh, sycl::read_only);

      cgh.parallel_for(sycl::range<2>(nx, ny), [=](sycl::id<2> id) {
        auto j = id[0] + 1;
        auto i = id[1] + 1;
        acc_curr[j][i] = acc_prev[j][i] + a * dt *
            ((acc_prev[j][i + 1] - 2.0 * acc_prev[j][i] + acc_prev[j][i - 1]) / dx2 +
             (acc_prev[j + 1][i] - 2.0 * acc_prev[j][i] + acc_prev[j - 1][i]) / dy2);
      });
    });
  }
}

void copy_to_buffer(sycl::queue Q, sycl::buffer<double, 2> buffer, const field* f)
{
    Q.submit([&](sycl::handler& h) {
    		auto acc = buffer.get_access<sycl::access::mode::write>(h);
    		h.copy(f->data.data(), acc);
    	});
}

void copy_from_buffer(sycl::queue Q, sycl::buffer<double, 2> buffer, field *f)
{
    Q.submit([&](sycl::handler& h) {
    		auto acc = buffer.get_access<sycl::access::mode::read>(h);
    		h.copy(acc, f->data.data());
    	}).wait();
}
