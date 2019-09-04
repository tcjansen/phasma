PROGRAM beerfit
use linminmod

implicit none
      
 INTEGER :: i
 INTEGER :: ndata
 INTEGER, PARAMETER :: kmax = 2
 REAL(8) :: omega0, period, ndatareal
 REAL(8), DIMENSION(:), ALLOCATABLE :: tdata, ydata
 REAL(8), PARAMETER :: pi = 3.141592653589793D0
 REAL(8), DIMENSION(2*kmax+1) :: beta

 read(*,*) ndatareal,period
 ndata = NINT(ndatareal)
 !write(*,*) ndata,period
 ALLOCATE(tdata(ndata))
 ALLOCATE(ydata(ndata))

 ! data should have transits masked and outliers removed already
 open(101,FILE='data.dat',FORM='FORMATTED',STATUS='UNKNOWN')
 DO i=1,ndata
   read(101,*) tdata(i),ydata(i)
 END DO
 close(101)
 
 omega0 = 2.0D0*pi/(period)
 
 call linmin(ndata,kmax*2+1,tdata,ydata,omega0,beta)
 
 write(*,*) beta(:)

END PROGRAM beerfit
