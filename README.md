# XRBinaryMCMC
Use ELR's XRBinary code to model a light curve, and use DFM's emcee code to find the best model.


# Starting
Run with `Model_XRbinary.py`, make sure you input a file with Phase, Magnitude (or luminosity), and corresponding error bars. If the input is in Magnitude, convert to luminosity using the convert_to_lum() function towards the end of the script.

Some things to keep in mind before using it, since some things are hard-corded in. In this version of the code the orbital period and semi-major velocity amplitude are fixed parameters.

The parameters being fit for are:
- Disk Luminosity
- Disk Eccentricity
- Disk Argument of Periastron
- Disk Size
- Disk Height
- Phase Offset
- Orbital Inclination
- Secondary Temperature
- Mass Ratio
- Edge Temperature

If this is not what you want, you can follow the format in the script and add or remove the parameters you want (or don't want).

# XRbinary
Also keep in mind the `star2tiles` and `disktiles` parameters, which by default are set to a low value. Set to the highest value your computational resources will allow.

# Plotting
The `Plot_XRbinary.py` script will generate corresponding trace and corner plots. Remember to modify it accordingly if you change the parameters being fit for in `Model_XRbinary.py`
