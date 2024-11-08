#include <iostream>
#include <mpi.h>
#include <omp.h>
#include <string>
#include <vector>

int main(int argc, char *argv[]) {
  // Initialize MPI communication.
  MPI_Init(&argc, &argv);

  // Identify the ID rank (process).
  int myid, nproc;
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  // Get number of active processes.
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  // Split the world communicator into subgroups.
  MPI_Comm host_comm;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                      &host_comm);
  int host_rank;
  MPI_Comm_rank(host_comm, &host_rank);

  // Get the node name.
  int name_len;
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  MPI_Get_processor_name(processor_name, &name_len);

  int myDevice = host_rank;
  // Set the device number to use in device constructs.
  omp_set_default_device(myDevice);

  // Return the number of devices available for offloading.
  int numDevice = omp_get_num_devices();

  if (myid == 0) {
    std::cout << std::endl;
    std::cout << "--nbr of MPI processes: " << nproc << std::endl;
    std::cout << "--nbr of gpus on each node: " << numDevice << std::endl;
    std::cout << "--nbr of nodes: " << nproc / numDevice << std::endl;
    std::cout << std::endl;
  }

  std::cout << "MPI-rank " << myid << " - Node " << processor_name << " - GPU_ID "
            << myDevice << " - GPUs-per-node " << numDevice << std::endl;

  // Finalize MPI.
  MPI_Finalize();

  return 0;
}
