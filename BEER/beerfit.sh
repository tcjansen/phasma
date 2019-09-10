#!/bin/bash
nlines=`cat data.dat | wc -l`
nlines=$(($nlines+1))
period='2.8493814'
echo $nlines $period | ./beerfit