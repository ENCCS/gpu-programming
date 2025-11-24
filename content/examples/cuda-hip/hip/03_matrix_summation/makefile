SRC=$(wildcard *.cpp)
PROGRAMS=$(SRC:%.cpp=%)

CC=hipcc
CFLAGS=-O2 --offload-arch=gfx90a

all: help
all-examples: $(PROGRAMS)

%: %.cpp
	$(CC) $(CFLAGS) $< -o $@

clean-all:
	rm -f $(PROGRAMS)

help:
	@grep '^#help:' $(firstword $(MAKEFILE_LIST))

#help: On LUMI
#help: ml craype-accel-amd-gfx90a rocm/6.0.3 PrgEnv-cray/8.5.0
#help:
#help: make 01_GPU_grid_block_thread_info          :
#help: make 02_matrix_thread_index_info            :
#help: make 03_matrix_summation_GPU_2D2D_2D1D_1D1D :
#help: make all-examples                           :
