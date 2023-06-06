// (c) 2023 ENCCS, CSC and the contributors
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "heat.h"
#include "pngwriter.h"

// Print out a picture of the temperature distribution
void field_write(field *heat, int iter)
{
#if HAVE_PNG
    char filename[64];

    // The actual write routine takes only the actual data
    // (without boundary layers) so we need to copy an array with that.
    std::vector<double> inner_data(heat->nx * heat->ny);
    auto inner_it = inner_data.begin();
    auto row_begin = heat->data.begin() + (heat->ny + 2) + 1;
    for (int i = 0; i < heat->nx; i++) {
        auto row_end = row_begin + heat->ny;
        std::copy(row_begin, row_end, inner_it);
        inner_it += heat->ny;
        row_begin = row_end + 2;
    }

    // Write out the data to a png file
    sprintf(filename, "%s_%04d.png", "heat", iter);
    save_png(inner_data.data(), heat->nx, heat->ny, filename, 'c');
#endif //HAVE_PNG
}
