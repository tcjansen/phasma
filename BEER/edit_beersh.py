import sys
import numpy as np

ID = sys.argv[1]
period = sys.argv[2]

with open('beerfit.sh', 'w') as beer:
	beer.write('#!/bin/bash' + '\n')
	beer.write('nlines=`cat data.dat | wc -l`' + '\n')
	beer.write('nlines=$(($nlines+1))' + '\n')
	beer.write("period='" + str(period) + "'\n")
	beer.write('echo $nlines $period | ./beerfit')
	beer.close()