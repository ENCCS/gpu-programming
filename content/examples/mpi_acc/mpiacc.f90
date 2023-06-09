program mpiacc

    use mpi
    use openacc
    use iso_fortran_env
    
    implicit none
    integer status(MPI_STATUS_SIZE)
    integer :: myid,ierr,nproc,tag
    integer :: host_rank,host_comm
    integer :: myDevice,numDevice
    integer :: resulten, ierror
    integer(int32) :: k, np
    integer(int32), parameter :: n=8192
    real(real64)   :: SumToT
    real(real64), allocatable :: f(:),f_send(:)
    character*(MPI_MAX_PROCESSOR_NAME) :: name

    call MPI_Init(ierr)       
    call MPI_COMM_RANK( MPI_COMM_WORLD, myid, ierr )        
    call MPI_COMM_SIZE( MPI_COMM_WORLD, nproc, ierr )

    call MPI_COMM_SPLIT_TYPE(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,MPI_INFO_NULL, host_comm,ierr)
    call MPI_COMM_RANK(host_comm, host_rank,ierr)
    
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
    
    if (mod(n,nproc).ne.0) then
        if (myid.eq.0) write(*,*) 'nproc has to divide n'
        stop
    endif

    np = n/nproc; tag=2023

    allocate(f(np))

    if(myid.eq.0) then
        allocate(f_send(n))
        CALL RANDOM_NUMBER(f_send)
    endif

    call MPI_Scatter(f_send,np,MPI_DOUBLE_PRECISION,f, np,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD, ierr)

    !offload f to GPUs
    !$acc enter data copyin(f)

    !update f: copy f from GPU to CPU
    !$acc update host(f)

    if(myid.lt.nproc-1) then
        call MPI_Send(f(np:np),1,MPI_DOUBLE_PRECISION,myid+1,tag,MPI_COMM_WORLD, ierr)
    endif

    if(myid.gt.0) then
        call MPI_Recv(f(1),1,MPI_DOUBLE_PRECISION,myid-1,tag,MPI_COMM_WORLD, status,ierr)
    endif

    !update f: copy f from CPU to GPU
    !$acc update device(f)

    !do something .e.g.
    !$acc kernels
    f = f/2.
    !$acc end kernels

    SumToT=0d0
    !$acc parallel loop reduction(+:SumToT)
    do k=1,np
        SumToT = SumToT + f(k)
    enddo
    !$acc end parallel loop

    !SumToT is by default copied back to the CPU
    call MPI_ALLREDUCE(MPI_IN_PLACE,SumToT,1,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD,ierr )

    !$acc exit data delete(f)

    if(myid.eq.0)  then
        print*,""
        print*,'--sum accelerated: ', real(sumToT)
        print*,""
    endif

    call MPI_FINALIZE( ierr )

end
