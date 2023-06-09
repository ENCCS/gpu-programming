         program laplace_gpuaware

           use mpi
           use openacc

           implicit none
            integer status(MPI_STATUS_SIZE)
            integer :: myid,ierr,nproc,nxp,nyp,nsend,tag1,tag2,tag
            integer :: i,j,k,iter
            integer :: host_rank,host_comm
            integer :: myDevice,numDevice
            integer :: resulten, ierror
            integer, parameter :: nx=2*4096,ny=nx
            integer, parameter :: max_iter=2000
            double precision, parameter    :: pi=4d0*datan(1d0)
            real, parameter    :: error=0.04
            double precision               :: max_err,d2fx,d2fy
            double precision, allocatable :: f(:,:),f_k(:,:)
            double precision, allocatable :: f_send(:,:)
            character*(MPI_MAX_PROCESSOR_NAME) :: name

! Initialise MPI communication      
        call MPI_Init(ierr)
! Identify the ID rank (process)        
        call MPI_COMM_RANK( MPI_COMM_WORLD, myid, ierr )
! Get number of active processes (from 0 to nproc-1)        
        call MPI_COMM_SIZE( MPI_COMM_WORLD, nproc, ierr )

        if(myid.eq.0) then
          print*, ""
          print*, "--GPU-aware MPI with OpenACC: 2D-Laplace Eq.--"
          print*, ""
          print*, "--Number of points nx= ",nx, " ny= ",ny
          print*, ""
       endif
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

!        write(*,'(A,I3,A,A,A,I3,A,I3)') "MPI-rank ", myid, " - Node ", trim(name), " - GPU_ID ", myDevice, " - GPUs-per-node ", numDevice


        if (mod(nx,nproc).ne.0) then
           if (myid.eq.0) write(*,*) 'nproc has to divide nx'
           stop
        else
           nxp = nx/nproc
        endif
        if (mod(ny,nproc).ne.0) then
           if (myid.eq.0) write(*,*) 'nproc has to divide ny'
           stop
        else
           nyp = ny/nproc
        endif

!Generate the Initial Conditions (ICs)
!Distribute the ICs over all processes using the operation MPI_Scatter
     allocate(f(0:nx+1,0:nyp+1));

     f=0d0; tag1=2020; tag2=2021

     if(myid.eq.0) then
       allocate(f_send(1:nx,1:ny))
        CALL RANDOM_NUMBER(f_send)
      endif

      call MPI_Scatter(f_send,nx*nyp,MPI_DOUBLE_PRECISION,&
                      f(1:nx,1:nyp), nx*nyp,MPI_DOUBLE_PRECISION,&
                      0,MPI_COMM_WORLD, ierr)

      call MPI_Barrier(MPI_COMM_WORLD, ierr)

      if(myid.eq.0) deallocate(f_send)

      allocate(f_k(1:nx,1:nyp))

       iter = 0; max_err = 1.0

       if(myid.eq.0) then
         print*,""
         print*, "--Start iterations",iter
         print*,""
       endif

!Unstructed data locality
!$acc enter data copyin(f) create(f_k)
       do while (max_err.gt.error.and.iter.le.max_iter)

!Device pointer f will be passed to MPI_send & MPI_recv       
!$acc host_data use_device(f)

!transfer the data at the boundaries to the neighbouring MPI-process
!send f(:,nyp) from myid-1 to be stored in f(:,0) in myid+1
         if(myid.lt.nproc-1) then
          call MPI_Send(f(:,nyp),(nx+2)*1,MPI_DOUBLE_PRECISION,myid+1,tag1,&
                       MPI_COMM_WORLD, ierr)
         endif

!receive f(:,0) from myid-1
         if(myid.gt.0) then
          call MPI_Recv(f(:,0),(nx+2)*1,MPI_DOUBLE_PRECISION,myid-1, &
                      tag1,MPI_COMM_WORLD, status,ierr)
         endif

!send f(:,1) from myid+1 to be stored in f(:,nyp+1) in myid-1
         if(myid.gt.0) then
          call MPI_Send(f(:,1),(nx+2)*1,MPI_DOUBLE_PRECISION,myid-1,tag2,&
                       MPI_COMM_WORLD, ierr)
         endif

!receive f(:,npy+1) from myid-1
        if(myid.lt.nproc-1) then
         call MPI_Recv(f(:,nyp+1),(nx+2)*1,MPI_DOUBLE_PRECISION,myid+1,&
                      tag2,MPI_COMM_WORLD, status,ierr)
        endif

!$acc end host_data

!$acc parallel loop present(f,f_k) collapse(2)
        do j=1,nyp
            do i=1,nx
               d2fx = f(i+1,j) + f(i-1,j)
               d2fy = f(i,j+1) + f(i,j-1)
               f_k(i,j) = 0.25*(d2fx + d2fy)
             enddo
          enddo
!$acc end parallel loop

          max_err=0.

!$acc parallel loop present(f,f_k) collapse(2) & 
!$acc reduction(max:max_err)
          do j=1,nyp
            do i=1,nx
               max_err = max(dabs(f_k(i,j) - f(i,j)),max_err)
               f(i,j) = f_k(i,j)
            enddo
          enddo
!$acc end parallel loop

!max_err is copied back to the CPU-host by default
!$acc data copy(max_err)
!$acc host_data use_device(max_err)
         call MPI_ALLREDUCE(MPI_IN_PLACE,max_err,1,&
              MPI_DOUBLE_PRECISION,MPI_MAX, MPI_COMM_WORLD,ierr )
!$acc end host_data
!$acc end data

          if(myid.eq.0) then
            if(mod(iter,50).eq.0 )write(*,'(i5,f10.6)')iter,max_err
          endif

          iter = iter + 1

        enddo
!$acc exit data copyout(f_k) delete(f)        

       call MPI_Barrier(MPI_COMM_WORLD, ierr)

       if(myid.eq.0) then
         print*, ''
         print*, '--Job is completed successfully--'
         print*,''
       endif

        call MPI_FINALIZE( ierr )
        end
 
