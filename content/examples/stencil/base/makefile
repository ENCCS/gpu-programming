OBJS=io.o setup.o utilities.o

CC=CC 
CFLAGS=-O3 
OMP=-fopenmp  

PROG_DATA=stencil_data
PROG_OFF=stencil_off
PROG=stencil

all: $(PROG) $(PROG_OFF) $(PROG_DATA)

$(PROG): $(OBJS) pngwriter.o core.o main.o
	@echo link $@ ...
	$(CC) $(CFLAGS) $(OMP) -o $@ $^

$(PROG_OFF): $(OBJS) pngwriter.o core-off.o main.o
	@echo link $@ ...
	$(CC) $(CFLAGS) $(OMP) -o $@ $^

$(PROG_DATA): $(OBJS) pngwriter.o core-data.o main-data.o 
	@echo link $@ ...
	$(CC) $(CFLAGS) $(OMP) -o $@ $^

%.o: %.cpp
	$(CC) $(CFLAGS) $(OMP) -c $<

%-off.o: %-off.cpp
	$(CC) $(CFLAGS) $(OMP) -c $<

%-data.o: %-data.cpp
	$(CC) $(CFLAGS) $(OMP) -c $<

pngwriter.o: pngwriter.c
	$(CC) $(CFLAGS) $(OMP) -c $<

clean:
	rm -f $(PROG) $(PROG_OFF) $(PROG_DATA) *.o

# core.cpp
# core-off.cpp
# core-data.cpp
# heat.h
# io.cpp
# main.cpp
# main-data.cpp
# pngwriter.c
# pngwriter.h
# setup.cpp
# utilities.cpp
