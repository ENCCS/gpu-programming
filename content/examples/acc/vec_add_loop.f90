program main
  implicit none

  integer, parameter :: nx = 102400
  integer :: i

  double precision :: vecA(nx), vecB(nx), vecC(nx)

  do i = 1, nx
     vecA(i) = 1.0
     vecB(i) = 1.0
  end do

  !$acc parallel loop
  do i = 1, nx
    vecC(i) = vecA(i) + vecB(i)
  end do
  !$acc end parallel loop

end program
