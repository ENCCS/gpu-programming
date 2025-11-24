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
#help: make 01_array_addition_cpu                  :
#help: make 02_array_addition_gpu_hip              :
#help: make 03_array_addition_deviceFunc_hip       :
#help: make all-examples                           :
