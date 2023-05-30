examples = assignDevice_omp laplace_mpiomp mpiomp mpiomp_gpuaware

FC=mpifort
FFLAGS=
OMP=-homp
ntask=1

all: help

all-examples: $(examples)

clean-mpiomp:
	rm -f $(@:clean-%=%)

clean-mpiomp_gpuaware:
	rm -f $(@:clean-%=%)

clean-laplace_mpiomp:
	rm -f $(@:clean-%=%)

clean-assignDevice_omp:
	rm -f $(@:clean-%=%)

clean-all: clean-mpiomp clean-laplace_mpiomp clean-mpiomp_gpuaware clean-assignDevice_omp


%: %.f90
	$(FC) $(FFLAGS) $(OMP) $< -o $@ 
	rm -f $(<:%.f90=%)_1.acc.*
	#export CRAY_ACC_DEBUG=3; ./$@

help:
	$(FC) --version
	@grep '^#help:' $(firstword $(MAKEFILE_LIST))

#help: On LUMI 
#help: ml craype-ompel-amd-gfx90a rocm/5.2.3 PrgEnv-cray/8.3.3
#help: 
#help: make mpiomp                         : fortran examples for mpi with openomp
#help: make mpiomp_gpuaware                : fortran examples for mpi with 
#help: make laplace_mpiomp                 :
#help: make assignDevice_omp               :
#help: make all-examples                   : build all examples
