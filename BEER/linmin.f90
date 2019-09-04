MODULE linminmod

implicit none
      
CONTAINS

!=======================================================================
SUBROUTINE linmin(nrows,ncols,t,y,omega0,beta)
         
 implicit none

 INTEGER :: i, j, r, c, iorder
 ! inputs
 REAL(8), INTENT(IN) :: omega0 ! =2pi/(2B) usually
 INTEGER, INTENT(IN) :: nrows, ncols
 REAL(8), DIMENSION(nrows), INTENT(IN) :: t, y
 ! intermediate results
 REAL(8), DIMENSION(nrows,ncols) :: Xarray
 REAL(8), DIMENSION(ncols,ncols) :: M, Minv
 REAL(8), DIMENSION(nrows) :: temp
 ! LAPACK variables
 INTEGER, DIMENSION(ncols) :: ipiv
 REAL(8), DIMENSION(:), ALLOCATABLE :: work
 INTEGER :: info1, info2, info3, lwork
 REAL(8) :: dblscalar
 ! output
 REAL(8), DIMENSION(ncols), INTENT(OUT) :: beta

 ! Create the X array
 DO c=1,ncols
	 DO r=1,nrows
		 Xarray(r,c) = basisfn(c,omega0*t(r))
	 END DO
 END DO
 
 ! Compute X^T.X
 DO i=1,ncols ! col
	 DO j=1,ncols ! row
		 ! Ok we're doing M(i,j)
		 M(i,j) = 0.0D0
		 DO r=1,nrows
			 M(i,j) = M(i,j) + Xarray(r,i)*Xarray(r,j)
		 END DO
	 END DO
 END DO

 ! Call invert... (note that M is symmetric)
 Minv = M
 call DGETRF(ncols,ncols,Minv,ncols,ipiv,info1)
 call DGETRI(ncols,Minv,ncols,ipiv,dblscalar,-1,info2)
 lwork = INT(dblscalar)
 ALLOCATE( work(lwork) )
 call DGETRI(ncols,Minv,ncols,ipiv,work,lwork,info3)
 !IF( info1 .NE. 0 .OR. info2 .NE. 0 .OR. info3 .NE. 0 ) THEN
 !  write(*,*) 'Inverse errors ',info1,info2,info3,M
 !END IF
 
 ! Now do Minv.X^T
 DO i=1,ncols
	 DO r=1,nrows
		 ! ok let's calculate temp(r)
		 temp(r) = 0.0D0
		 DO c=1,ncols
			 temp(r) = temp(r) + Minv(i,c)*Xarray(r,c)
		 END DO
		 temp(r) = y(r)*temp(r)
	 END DO
	 beta(i) = SUM(temp(:))
 END DO
 
END SUBROUTINE linmin
!=======================================================================

!=======================================================================
FUNCTION basisfn(i,x)
	
	implicit none
	
	INTEGER, INTENT(IN) :: i
	REAL(8), INTENT(IN) :: x
	INTEGER :: k
	REAL(8) :: basisfn
	
	k = FLOOR(0.5*i)
	IF( i .EQ. 1 ) THEN
	  basisfn = 1.0D0
  ELSE IF( MOD(i,2) .EQ. 0 ) THEN
	  ! c is odd => sine functions
	  basisfn = DSIN(k*x)
	ELSE
		basisfn = DCOS(k*x)
	END IF
	
END FUNCTION
!=======================================================================

END MODULE linminmod
