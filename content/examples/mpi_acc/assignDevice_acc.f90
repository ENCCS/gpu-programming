program assignDevice

    use mpi
    use openacc

    implicit none
    integer status(MPI_STATUS_SIZE)
    integer :: myid,ierr,nproc
    integer :: host_rank,host_comm
    integer :: myDevice,numDevice
    integer :: resulten, ierror
    character(len=300) :: env_var
    character*(MPI_MAX_PROCESSOR_NAME) :: name

    ! Initialise MPI communication      
    call MPI_Init(ierr)
    ! Identify the ID rank (process)        
    call MPI_COMM_RANK( MPI_COMM_WORLD, myid, ierr )
    ! Get number of active processes (from 0 to nproc-1)        
    call MPI_COMM_SIZE( MPI_COMM_WORLD, nproc, ierr )

    ! Split the world communicator into subgroups of commu, each of which
    ! contains processes that run on the same node, and which can create a
    ! shared memory region (via the type MPI_COMM_TYPE_SHARED).
    ! The call returns a new communicator "host_comm", which is created by
    ! each subgroup.        
    call MPI_COMM_SPLIT_TYPE(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,&
                            MPI_INFO_NULL, host_comm,ierr)
    call MPI_COMM_RANK(host_comm, host_rank,ierr)

    ! Gets the node name
    call MPI_GET_PROCESSOR_NAME(name, resulten, ierror)

    myDevice = host_rank

    ! Sets the device number and the device type to be used
    call acc_set_device_num(myDevice, acc_get_device_type())

    ! Returns the number of devices available on the host
    numDevice = acc_get_num_devices(acc_get_device_type())

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
