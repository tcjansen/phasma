#!/bin/bash

tic=38846515

period=2.8493814

duration=3.80088

ephemeris=1426.4732900001109

python make_data-dat.py $tic $period $duration $ephemeris
python edit_beersh.py $tic $period
./make.sh
values=`./beerfit.sh`
coefffile=`echo "../"$tic"/coefficients.txt"`
echo $values
echo $values > $coefffile
