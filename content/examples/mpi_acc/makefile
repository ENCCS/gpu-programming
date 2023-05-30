examples = assignDevice_acc laplace_mpiacc mpiacc mpiacc_gpuaware

FC=mpifort
FFLAGS=
OMP=-hacc
ntask=1

all: help

all-examples: $(examples)

clean-mpiacc:
	rm -f $(@:clean-%=%)

clean-mpiacc_gpuaware:
	rm -f $(@:clean-%=%)

clean-laplace_mpiacc:
	rm -f $(@:clean-%=%)

clean-assignDevice_acc:
	rm -f $(@:clean-%=%)

clean-all: clean-mpiacc clean-laplace_mpiacc clean-mpiacc_gpuaware clean-assignDevice_acc


%: %.f90
	$(FC) $(FFLAGS) $(OMP) $< -o $@ 
	rm -f $(<:%.f90=%)_1.acc.*
	#export CRAY_ACC_DEBUG=3; ./$@

help:
	$(FC) --version
	@grep '^#help:' $(firstword $(MAKEFILE_LIST))

#help: On LUMI 
#help: ml craype-accel-amd-gfx90a rocm/5.2.3 PrgEnv-cray/8.3.3
#help: 
#help: make mpiacc                         : fortran examples for mpi with openacc
#help: make mpiacc_gpuaware                : fortran examples for mpi with 
#help: make laplace_mpiacc                 :
#help: make assignDevice_acc               :
#help: make all-examples                   : build all examples
