OBJS=io.o setup.o utilities.o

CC=hipcc
CFLAGS=-O3 
OMP=-fopenmp --offload-arch=gfx90a 
#OMP=-fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx90a 

PROG_DATA=stencil_data
PROG_OMP=stencil_omp
PROG=stencil

all: $(PROG) $(PROG_OMP) $(PROG_DATA)

$(PROG): $(OBJS) pngwriter.o core.o main.o
	@echo link $@ ...
	$(CC) $(CFLAGS) $(OMP) -o $@ $^

$(PROG_OMP): $(OBJS) pngwriter.o core-omp.o main.o
	@echo link $@ ...
	$(CC) $(CFLAGS) $(OMP) -o $@ $^

$(PROG_DATA): $(OBJS) pngwriter.o core-data.o main-data.o 
	@echo link $@ ...
	$(CC) $(CFLAGS) $(OMP) -o $@ $^

%.o: %.cpp
	$(CC) $(CFLAGS) $(OMP) -c $<

%-omp.o: %-omp.cpp
	$(CC) $(CFLAGS) $(OMP) -c $<

%-data.o: %-data.cpp
	$(CC) $(CFLAGS) $(OMP) -c $<

pngwriter.o: pngwriter.c
	$(CC) $(CFLAGS) $(OMP) -c $<

clean:
	rm -f $(PROG) $(PROG_OMP) $(PROG_DATA) *.o

# core.cpp
# core-omp.cpp
# core-data.cpp
# heat.h
# io.cpp
# main.cpp
# main-data.cpp
# pngwriter.c
# pngwriter.h
# setup.cpp
# utilities.cpp
