from heat_core import *

# io.py + main.py

import time
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'BrBG'


def field_write(heat, iter):
    plt.gca().clear()
    plt.imshow(heat.data)
    plt.axis('off')
    plt.savefig('heat_{0:03d}.png'.format(iter))


def start_time (): return time.perf_counter()
def stop_time (): return time.perf_counter()


def main(args):
    current, previous, nsteps = initialize(args)

    # Output the initial field and its temperature
    field_write(current, 0)
    average_temp = field_average(current)
    print("Average temperature, start: %f\n" % average_temp)

    # Set diffusivity constant
    a = 0.5
    # Compute the largest stable time step
    dx2 = current.dx * current.dx
    dy2 = current.dy * current.dy
    dt = dx2 * dy2 / (2.0 * a * (dx2 + dy2))
    # Set output interval
    output_interval = 1500

    # Start timer
    start_clock = start_time()
    # Time evolution
    for iter in range(1,nsteps+1):
        evolve(current, previous, a, dt)
        if (iter % output_interval == 0):
            field_write(current, iter)
        # Swap current and previous fields for next iteration step
        current, previous = previous, current

    # Stop timer
    stop_clock = stop_time()

    # Output the final field and its temperature
    average_temp = field_average(previous)
    print("Average temperature at end: %f\n" % average_temp)
    # Compare temperature for reference
    if (args.rows == ROWS and args.cols == COLS and args.nsteps == NSTEPS):
        print("Control temperature at end: 59.281239\n")
    field_write(previous, nsteps)

    # Determine the computation time used for all the iterations
    print("Iterations took %.3f seconds.\n" % (stop_clock - start_clock))


if __name__ == '__main__':
    # Process command line arguments
    parser = argparse.ArgumentParser(description='Heat flow example')
    parser.add_argument('rows', type=int, nargs='?', default=ROWS,
                        help='number of grid rows')
    parser.add_argument('cols', type=int, nargs='?', default=COLS,
                        help='number of grid cols')
    parser.add_argument('nsteps', type=int, nargs='?', default=NSTEPS,
                        help='number of time steps')
    
    args = parser.parse_args()
    main(args)
    
