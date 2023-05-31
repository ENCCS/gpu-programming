program mpiomp

           use mpi
           use omp_lib
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

! Sets the device number to use in device constructs by setting the
! initial value of the default-device-var 
       call omp_set_default_device(myDevice)

! Returns the number of devices available for offloading.
       numDevice = omp_get_num_devices()

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

      if(myid.eq.0) deallocate(f_send)

!offload f to GPUs
!$omp target enter data device(myDevice) map(to:f)

!copy data from GPU to CPU
!$omp target update device(myDevice) from(f)

       if(myid.lt.nproc-1) then
          call MPI_Send(f(np:np),1,MPI_DOUBLE_PRECISION,myid+1,tag,MPI_COMM_WORLD, ierr)
        endif

        if(myid.gt.0) then
          call MPI_Recv(f(1),1,MPI_DOUBLE_PRECISION,myid-1,tag,MPI_COMM_WORLD, status,ierr)
        endif

!copy the updated data in CPU to GPU
!$omp target update device(myDevice) to(f)

!do something .e.g.
!$omp target teams distribute parallel do
       do k=1,np
          f(k) = f(k)/2.
       enddo
!$omp end target teams distribute parallel do

         SumToT=0d0
!$omp target teams distribute parallel do reduction(+:SumToT)
         do k=1,np
            SumToT = SumToT + f(k)
         enddo
!$omp end target teams distribute parallel do  

!SumToT is by default copied back to the CPU
         call MPI_ALLREDUCE(MPI_IN_PLACE,SumToT,1,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD,ierr )

!$omp target exit data map(delete:f)

        if(myid.eq.0)  then
          print*,""
          print*,'--sum accelerated: ', real(sumToT)
          print*,""
        endif

         call MPI_FINALIZE( ierr )

        end
