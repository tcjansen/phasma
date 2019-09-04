#!/bin/bash

tic=38846515

period=2.84938

duration=3.776111

ephemeris=1326.7450769999996

python make_data-dat.py $tic $period $duration $ephemeris
python edit_beersh.py $tic $period
./make.sh
values=`./beerfit.sh`
coefffile=`echo "../"$tic"/coefficients.txt"`
echo $values
echo $values > $coefffile
