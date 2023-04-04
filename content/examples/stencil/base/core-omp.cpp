#include "heat.h"

// Update the temperature values using five-point stencil
// Arguments:
//   curr: current temperature values
//   prev: temperature values from previous time step
//   a: diffusivity
//   dt: time step
void evolve(field *curr, field *prev, double a, double dt)
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
  
  // Offload value update to GPU target (fallback to CPU is possible)
  #pragma omp target teams distribute parallel for \
  map(currdata[0:(nx+2)*(ny+2)],prevdata[0:(nx+2)*(ny+2)])
  for (int i = 1; i < nx + 1; i++) {
    for (int j = 1; j < ny + 1; j++) {
      int ind = i * (ny + 2) + j;
      int ip = (i + 1) * (ny + 2) + j;
      int im = (i - 1) * (ny + 2) + j;
      int jp = i * (ny + 2) + j + 1;
      int jm = i * (ny + 2) + j - 1;
      currdata[ind] = prevdata[ind] + a*dt*
	    ((prevdata[ip] - 2.0*prevdata[ind] + prevdata[im]) / dx2 +
	     (prevdata[jp] - 2.0*prevdata[ind] + prevdata[jm]) / dy2);
    }
  }
}
