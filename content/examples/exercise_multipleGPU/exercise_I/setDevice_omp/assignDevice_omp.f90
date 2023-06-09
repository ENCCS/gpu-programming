         program assignDevice

           use mpi
           use omp_lib

           implicit none
            integer status(MPI_STATUS_SIZE)
            integer :: myid,ierr,nproc
            integer :: host_rank,host_comm
            integer :: myDevice,numDevice
            integer :: resulten, ierror
            character(len=300) :: env_var
            character*(MPI_MAX_PROCESSOR_NAME) :: name

        call MPI_Init(ierr)
        call MPI_COMM_RANK( MPI_COMM_WORLD, myid, ierr )
        call MPI_COMM_SIZE( MPI_COMM_WORLD, nproc, ierr )

        call MPI_COMM_SPLIT_TYPE(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,&
                               MPI_INFO_NULL, host_comm,ierr)
        call MPI_COMM_RANK(host_comm, host_rank,ierr)

        call MPI_GET_PROCESSOR_NAME(name, resulten, ierror)

        myDevice = host_rank

!QUESTION-1:        
! Incorporate the OpenMP function that enables to set the device number to be used in device constructs

        call .....

!QUESTION-2:      
! Incorporate the OpenMP function that enables to return the number of devices available for offloading

        numDevice =


        if(myid.eq.0) then
          print*,''
          print*, '--nbr of MPI processes: ', nproc
          print*, '--nbr of gpus on each node: ', numDevice
          print*, '--nbr of nodes: ', nproc/numDevice
          print*,''
        endif

        write(*,'(A,I3,A,A,A,I3,A,I3)') "MPI-rank ", myid, " - Node ", trim(name), " - GPU_ID ", myDevice, " - GPUs-per-node ", numDevice

        call MPI_FINALIZE( ierr )
        end

