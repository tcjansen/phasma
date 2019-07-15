How I want this to work:

There will be a folder titled 'phamsma', and inside are modules specific to 
classes:
--> licenses
	--> LICENSE.rst
--> docs
	--> index.rst  # has intro, installation and setup, getting started, etc
	--> phasma
		--> target.rst
		--> phasecurve.rst
		--> lightcurve.rst
--> phasma
	--> __init__.py
	--> target.py
	--> phasecurve.py
	--> lightcurve.py
	--> tests
		--> test_target.py
		--> test_phasecurve.py
		--> test_lightcurve.py

Then the usage would go like:

>>> from phasma.target import Target
>>> from phasma.phasecurve import Phasecurve
>>> from phasma.lightcurve import Lightcurve
>>>
>>> import astropy.constants as u
>>> from astropy.time import Time

define target parameters
>>> tic_id = '231663901'
>>> orbital_period = 1.43036763 * u.day
>>> transit_epoch = Time(2455392.31659, format='jd')  # approximate BJD
>>> transit_duration = 1.638765 * u.hr

make the target object for phasma
>>> target_object = Target(tic_id, orbital_period, transit_epoch, transit_duration)

then you could do things like
>>> target_object.tic_id  # can be used for convenient file naming, etc
231663901

make a phase curve object
>>> target_phasecurve = Phasecurve(target_object)

plot the phase curve
>>> target_phasecurve.plot

write the phase curve to a file
>>> target_phasecurve.write(filename='custom_name.fits')

fit to beer model
>>> target_phasecurve.fit(model=beer)
where beer is a function name such thatone can call their own function if they 
wish

fit to cosine model
>>> target_phasecurve.fit(model=cosine)

make a light curve object
>>> target_lightcurve = Lightcurve(target_object)

plot the light curve
>>> target_lightcurve.plot

write the light curve to a file
>>> target_lightcurve.write
