SRC_cu=$(wildcard *.cu) 01_array_addition_cpu.cpp
PROGAMS=$(SRC:%.cu=%) 01_array_addition_cpu

all: help

01_array_addition_cpu: 01_array_addition_cpu.cpp
	echo c++ $< -o $@

%: %.cu
	echo nvcc $< -o $@

clean-all:
	rm -f $(PROGAMS)

help:
	@grep '^#help:' $(firstword $(MAKEFILE_LIST))

#help: NSC tetralith
#help: ml 
#help: 
#help: make 01_array_addition_cpu             :
#help: make 02_array_addition_gpu             :
#help: make 03_array_addition_deviceFunc      :
