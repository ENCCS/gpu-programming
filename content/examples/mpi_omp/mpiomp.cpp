#include <cstdlib> // for std::exit
#include <iostream>
#include <mpi.h>
#include <omp.h>
#include <string>
#include <vector>

int main(int argc, char *argv[]) {
  const size_t n = 8192;
  const int tag = 2024;

  MPI_Init(&argc, &argv);
  int myid, nproc;
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  MPI_Comm host_comm;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                      &host_comm);
  int host_rank;
  MPI_Comm_rank(host_comm, &host_rank);
  int name_len;
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  MPI_Get_processor_name(processor_name, &name_len);

  int myDevice = host_rank;
  omp_set_default_device(myDevice);
  int numDevice = omp_get_num_devices();

  if (myid == 0) {
    std::cout << "\n";
    std::cout << "--nbr of MPI processes: " << nproc << "\n";
    std::cout << "--nbr of gpus on each node: " << numDevice << "\n";
    std::cout << "--nbr of nodes: " << nproc / numDevice << "\n";
    std::cout << "\n";
  }

  std::cout << "MPI-rank " << myid << " - Node " << processor_name
            << " - GPU_ID " << myDevice << " - GPUs-per-node " << numDevice
            << "\n";

  if (n % nproc != 0) {
    if (myid == 0)
      std::cout << "nproc has to divide n\n";
    return -1;
  }

  int np = n / nproc;
  std::vector<double> f_vector(np);
  double *f = f_vector.data(); // Need a raw pointer for OpenMP
  std::vector<double> f_send;

  if (myid == 0) {
    f_send.resize(n);
    for (int i = 0; i < n; i++) {
      f_send[i] = static_cast<double>(rand()) / RAND_MAX;
    }
  }

  MPI_Scatter(f_send.data(), np, MPI_DOUBLE, f, np, MPI_DOUBLE, 0,
              MPI_COMM_WORLD);

// Offload f to GPUs
#pragma omp target enter data map(to : f[0 : np]) device(myDevice)

  // update f: copy f from GPU to CPU
#pragma omp target update device(myDevice) from(f)

  if (myid < nproc - 1) {
    MPI_Send(&f[np - 1], 1, MPI_DOUBLE, myid + 1, tag, MPI_COMM_WORLD);
  }
  if (myid > 0) {
    MPI_Recv(&f[0], 1, MPI_DOUBLE, myid - 1, tag, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
  }

// update f: copy f from CPU to GPU
#pragma omp target update device(myDevice) to(f)

// Update, operate, and offload data back to GPUs
#pragma omp target teams distribute parallel for device(myDevice)
  for (int k = 0; k < np; ++k) {
    f[k] /= 2.0;
  }

  double SumToT = 0.0;

#pragma omp target teams distribute parallel for reduction(+ : SumToT)         \
    device(myDevice)
  for (int k = 0; k < np; ++k) {
    SumToT += f[k];
  }

  MPI_Allreduce(MPI_IN_PLACE, &SumToT, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

#pragma omp target exit data map(delete : f[0 : np])

  if (myid == 0) {
    std::cout << "\n";
    std::cout << "--sum accelerated: " << SumToT << "\n";
    std::cout << "\n";
  }

  MPI_Finalize();

  return 0;
}
