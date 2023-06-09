SRC_cu=$(wildcard *.cu)
PROGAMS=$(SRC:%.cu=%)

all: help

%: %.cu
	nvcc $< -O3 -o $@

clean-all:
	rm -f $(PROGAMS)

help:
	@grep '^#help:' $(firstword $(MAKEFILE_LIST))

#help: NSC tetralith
#help: ml 
#help: 
#help: make 01_array_reduce_cpu             :
#help: make 02_array_reduce_gpu             : 
#help: make 03_array_reduce_gpu_atomic      :
#help: make 04_array_reduce_gpu_shuffle     :
#help: make 05_array_reduce_gpu_parallelism :
#help: make 06_array_reduce_gpu_static      :

