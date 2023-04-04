#include <cstdlib>
#include <cassert>

#include "heat.h"

// Copy data on heat1 into heat2
void field_copy(field *heat1, field *heat2)
{
    assert(heat1->nx == heat2->nx);
    assert(heat1->ny == heat2->ny);
    assert(heat1->data.size() == heat2->data.size());
    std::copy(heat1->data.begin(), heat1->data.end(),
              heat2->data.begin());
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
    // Include also boundary layers
    int newSize = (heat->nx + 2) * (heat->ny + 2);
    heat->data.resize(newSize, 0.0);
}

// Calculate average temperature over the non-boundary grid cells
double field_average(field *heat)
{
     double average = 0.0;

     for (int i = 1; i < heat->nx + 1; i++) {
       for (int j = 1; j < heat->ny + 1; j++) {
         int ind = i * (heat->ny + 2) + j;
         average += heat->data[ind];
       }
     }

     average /= (heat->nx * heat->ny);
     return average;
}
