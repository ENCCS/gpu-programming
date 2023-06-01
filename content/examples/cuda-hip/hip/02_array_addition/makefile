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
#help: make 01_array_addition_cpu                  :
#help: make 02_array_addition_gpu_hip              :
#help: make 03_array_addition_deviceFunc_hip       :
#help: make all-examples -j                        :
#help: make clean-all                              :
