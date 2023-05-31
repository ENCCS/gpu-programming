examples = vec_add_kernels vec_add_loop
FC=ftn
FFLAGS=
ACC=-hacc

all: help

all-examples: $(examples)

clean-vec_add_loop:
	rm -f $(@:clean-%=%)
	
clean-vec_add_kernels:
	rm -f $(@:clean-%=%)

clean-all: clean-vec_add_loop clean-vec_add_kernels

%: %.f90
	$(FC) $(FFLAGS) $(ACC) $< -o $@ 
	rm -f $(<:%.f90=%)_1.acc.*
	# export CRAY_ACC_DEBUG=3; ./$@

help:
	@grep '^#help:' $(firstword $(MAKEFILE_LIST))

#help: On LUMI 
#help: ml craype-accel-amd-gfx90a rocm/5.2.3 PrgEnv-cray/8.3.3
#help: 
#help: make vec_add_kernels     : fortran examples openacc use kernel
#help: make vec_add_loop        : fortran examples openacc use loop
#help:
#help: make all-examples        : build all examples
