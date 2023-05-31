FC=ftn
FFLAGS=

CC=hipcc
CFLAGS=

OMP=-fopenmp

all: help

clean-vec_add_target_f_omp:
	rm -f $(@:clean-%=%)
	
clean-vec_add_target_f:
	rm -f $(@:clean-%=%)

clean-all: clean-vec_add_target_f_omp clean-vec_add_target_f

vec_add_target_f_omp: vec_add_target.f90
	$(FC) $(FFLAGS) $(OMP) $< -o $@ 
	rm -f $(<:%.f90=%)_1.acc.*
	export CRAY_ACC_DEBUG=3; ./$@

vec_add_target_f: vec_add_target.f90
	$(FC) $(FFLAGS) -o $@ $<
	export CRAY_ACC_DEBUG=3; ./$@


help:
	@grep '^#help:' $(firstword $(MAKEFILE_LIST))

#help: On LUMI 
#help: ml craype-accel-amd-gfx90a rocm/5.2.3 PrgEnv-cray/8.3.3
#help: 
#help: make vec_add_target_f               : fortran examples without openmp offload
#help: make vec_add_target_f_omp           : fortran examples with openmp offload
#help:
#help:
