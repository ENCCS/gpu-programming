SRC_cu=$(wildcard *.cu) 01_hello.c
PROGAMS=$(SRC:%.cu=%) 01_hello

all: help

01_hello: 01_hello.c
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
#help: make 01_hello  : compile
#help: make 02_hello  : compile
#help: make 03_hello  : compile
#help: make 04_hello  : compile
#help: make 05_hello  : compile
