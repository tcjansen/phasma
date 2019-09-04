import subprocess

with open('targetlist.txt') as tl:
	for tic in tl.readlines()[1:]:
		with open('BEER/run.sh', 'w') as run:
			run.write('#!/bin/bash' + '\n\n')

			run.write('tic=' + str(tic) + '\n\n')

			run.write('python make_data-dat.py $tic' + '\n')
			run.write('python edit_beersh.py $tic' + '\n')
			run.write('./make.sh' + '\n')
			run.write('values=`./beerfit.sh`' + '\n')
			run.write('coefffile=`echo "../targets/"$tic"/coefficients.txt"`' + '\n')
			run.write('echo $values' + '\n')
			run.write('echo $values > $coefffile' + '\n')
			run.write('python plot_fit.py $tic' + '\n')

			run.close()

subprocess.call(['BEER/run.sh'])