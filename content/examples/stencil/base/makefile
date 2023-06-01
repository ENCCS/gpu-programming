OBJS=io.o main.o pngwriter.o setup.o utilities.o

CC=hipcc
CFLAGS=-O3 
OMP=-fopenmp --offload-arch=gfx90a

PROG_OMP=stencil_omp
PROG=stencil

all: $(PROG) $(PROG_OMP)

$(PROG): $(OBJS) pngwriter.o core.o
	@echo link $@ ...
	$(CC) $(CFLAGS) $(OMP) -o $@ $^

$(PROG_OMP): $(OBJS) pngwriter.o core-omp.o
	@echo link $@ ...
	$(CC) $(CFLAGS) $(OMP) -o $@ $^

%.o: %.cpp
	$(CC) $(CFLAGS) $(OMP) -c $<

%-omp.o: %-omp.cpp
	$(CC) $(CFLAGS) $(OMP) -c $<

pngwriter.o: pngwriter.c
	$(CC) $(CFLAGS) -c $<

clean:
	rm -f $(PROG) $(PROG_OMP) *.o

# core.cpp
# core-omp.cpp
# heat.h
# io.cpp
# main.cpp
# pngwriter.c
# pngwriter.h
# setup.cpp
# utilities.cpp
