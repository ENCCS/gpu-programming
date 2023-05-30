SRC=$(wildcard *.cpp)
PROGAMS=$(SRC:%.cpp=%)

CC=hipcc
CFLAGS=--amdgpu-target=gfx90a

all: help
all-examples: $(PROGAMS)

%: %.cpp
	$(CC) $(CFLAGS) $< -o $@

clean-all:
	rm -f $(PROGAMS)

help:
	@grep '^#help:' $(firstword $(MAKEFILE_LIST))

#help: On LUMI 
#help: ml craype-accel-amd-gfx90a rocm/5.2.3 PrgEnv-cray/8.3.3
#help: 
#help: make 01_GPU_grid_block_thread_info          :
#help: make 02_matrix_thread_index_info            :
#help: make 03_matrix_summation_GPU_2D2D_2D1D_1D1D :
#help: make all-examples                           :
