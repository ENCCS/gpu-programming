SRC_cu=$(wildcard *.cu)
PROGAMS=$(SRC:%.cu=%)

all: help

%: %.cu
	echo nvcc $< -o $@

clean-all:
	rm -f $(PROGAMS)

help:
	@grep '^#help:' $(firstword $(MAKEFILE_LIST))

#help: NSC tetralith
#help: ml 
#help: 
#help: make 01_GPU_grid_block_thread_info          :
#help: make 02_matrix_thread_index_info            :
#help: make 03_matrix_summation_GPU_2D2D_2D1D_1D1D :
