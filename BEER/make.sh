#!/bin/bash
#

rm -f *.o *.mod
rm -f beerfit
gfortran -O3 -llapack -c linmin.f90 && gfortran -O3 -llapack -o beerfit beerfit.f90 linmin.o
