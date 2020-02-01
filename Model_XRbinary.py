import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
import datetime
import subprocess
from scipy.interpolate import interp1d
import platform
Start = datetime.datetime.now()

################################
########## PARAMETERS ##########
################################

# Input data file in 3 column format (Phase, Magnitude or Luminosity, Error)
# If the input is in magnitude use convert_to_lum() function
File_name = 'lc_gkper_mod.txt'

# Specify the band of the data : U, B, V, R, I, J, H, K
band = 'R'

### Fixed Parameters ###

# Orbital Period of the system in days
period = 1.9968
# Semi-major velocity amplitude
k2_center = 126.0

### Flat or log Prior Parameters ###

# Absolute prior range for each parameter
min_10maindiskt   , max_10maindiskt   =   31.0, 39.0     # 10^Disk Luminosity in erg/s
min_diske         , max_diske         =    0.0, 0.1      # Eccentricity of the Accretion Disk
min_diskzetazero  , max_diskzetazero  =    0.0, 360.0    # Argument of Periastron
min_disk_size     , max_disk_size     =    0.0, 0.9      # Size of the disk in units of semi-major axis
min_vsini         , max_vsini         =     20, 250      # rotational velocity of the star v * sin(i)
min_diskh         , max_diskh         =  0.005, 0.1      # Height of the disk in scale units
min_m2            , max_m2            =    0.0, 3.0      # Secondary mass
min_phaseoffset   , max_phaseoffset   =   -0.1, 0.1      # Phase offset
min_inclination   , max_inclination   =     10, 89       # orbital inclination
min_star2temp     , max_star2temp     =   1000, 10000    # Secondary temperature
min_massratio     , max_massratio     =   0.01, 0.99     # Mass Ratio
min_edge_T        , max_edge_T        =    500, 5000     # Temperature of accretion disk edge

### Gaussian Prior Parameters ###

# Center and error bars for vsini, assuming gaussian prior
vsini_center = 107.0
width_vsini  = 5.0

### Other Parameters for XRbinary ###

# List of the rest of the parameters
star1lum        = 2.2E15 # Luminosity of compact object in erg/s
star1temp       = 3000   # Temperature of compact object in K
star2albedo     = 0.5    # Albedo of donor
diskalbedo      = 1.0    # Albedo of the disk
r_disk          = 0.30   # Radius of the disk in units of a (Will be overwritten in this version)
r_isco          = 0.001  # Radius of the ISCO in units of a
disk_h_power    = 1.5    # Exponent power of the disk height
edge_T_spot     = 1.0    # Temperature of a spot on the edge of the disk
edge_spot_angle = 180.0  # Position of the spot on the edge of the disk
edge_spot_width = 20.0   # Width of the spot on the edge of the disk

# Fixed code parameters
min_phase       = 0.0    # Minimum Light curve phase
max_phase       = 1.0    # Maximum light curve phase
step_size       = 0.01   # Step size of light curve
star2tiles      = 3000   # Number of tiles for donor
disktiles       = 3000   # Number of tiles in the accretion disk

# Gravitational Constant
G = 1.327E11 # in km^3 / (Solar masses * seconds ^ 2)

# Import Lightcurve Data
Phase_data_a, Light_data_a, Error_data_a = np.genfromtxt(File_name, unpack = True)

# Specify the distance to the star in kpc (Irrelevant if normalization is turned ON)
d_star = 5.0

# MCMC Parameters
n_steps   = 10 # Number of steps
n_walkers = 40 # Number of walkers (must be even and > x2 number of parameters)
n_cores   = 4  # Number of cores to use for computation

# Empty Variables for future use
itera = 0
RandomNumber = np.int(np.random.random()*100000000000)

###############################
########## FUNCTIONS ##########
###############################

def gaussian(x, c, w):
    return np.exp(-(x - c)**2/(2*w**2))

def create_dists(n, dist_list):
    '''
    Given a number of walkers and type of distribution, the function returns
    samples from the desired distribution for each parameter.

    Parameters
    ----------
    n: Number of walkers
    dist_list: tupple list of type ['dist', 0, 1]:

    Examples
    --------
    Gaussian: ['g', mean, stddev]
    Uniform:  ['u', low_lim, up_lim]
    Jeffrys:  ['j', low_lim, up_lim]
    Gaussian Jeffrys:  ['jg', low_lim, up_lim]
    Truncated Gaussian:
        lower truncated: ['ltg', mean, stddev, lower_lim]
        upper truncated: ['utg', mean, stddev, upper_lim]
        both truncated: ['btg', mean, stddev, lower_lim, upper_lim]

    Returns
    -------
    np.array: A distribution of dimension n with random values picked
              in the distribution specified.
    '''

    def create_tg(mean, stddev, low_lim=-np.inf, up_lim=np.inf):
        tg = []
        # continue generating new gaussian values until enough are w/in limits
        while len(tg) < n:
            new_g = (mean+stddev*np.random.randn(n-len(tg))).tolist()
            new_tg = [elt for elt in new_g if elt > low_lim and elt < up_lim]
            [tg.append(elt) for elt in new_tg]
        return np.array(tg)
     
    dists = []
    for dst in dist_list:
        if dst[0] == 'g':
            dists.append(dst[1]+dst[2]*np.random.randn(n))
        elif dst[0] == 'u':
            dists.append(dst[1]+(dst[2]-dst[1])*np.random.rand(n))
        elif dst[0] == 'j':
            dists.append(10**np.random.uniform(dst[1], dst[2], n))
        elif dst[0] == 'jg':
            dists.append(10**np.random.normal(dst[1], dst[2], n))
        elif dst[0] == 'ltg':
            dists.append(create_tg(dst[1],dst[2],low_lim=dst[3]))
        elif dst[0] == 'utg':
            dists.append(create_tg(dst[1],dst[2], up_lim=dst[3]))
        elif dst[0] == 'btg':
            dists.append(create_tg(dst[1],dst[2],low_lim=dst[3],up_lim=dst[4]))
        else:
            print("first element of list must be "),
            print("\'g\',\'u\',\'ltg\',\'utg\',or \'btg\'")
    return np.swapaxes(dists,0,1)

# Function to create the paramter file that XRbinary will read
def create_parfile(ParFile, verbose = "ON", diagnostics = "OFF 0.25 R", star1 = "ON", star2 = "ON", star2spots = "OFF", disk = "ON",
                   diskrim = "OFF", disktorus = "OFF", innerdisk = "OFF", diskspots = "OFF", ADC = "OFF", thirdlight = "OFF",
                   irradiation = "ON", min_phase = "0.0", max_phase = "1.0", step_size = "0.01", phaseoffset = "0.00", band = "I",
                   period = "0.05821", m1 = "4.0", massratio = "0.3", inclination = "23.0", star1lum = "2.2E13", star1temp = "3000",
                   star2tiles = "10000", star2temp = "1500", star2albedo = "0.5", disktiles = "10000", diske = "0.0",
                   diskzetazero = "30.0", diskalbedo = "1.0", r_isco = "0.001", r_disk = "0.30", disk_h = "0.02", disk_h_power = "1.2",
                   maindiskt = "1.0E37", edge_T = "5000.0", edge_T_spot = "15000.0", edge_spot_angle = "110.0", edge_spot_width = "20.0",
                   File_name = "J1407out.txt"):
    '''
    Function will create a parameter file from a set of default parameters for the XRbinary code to read in.
    The list of parameters and their description are below:
    ----------
    Output is a parameter file with the name specified by the variable "ParFile"
    '''

    output = ["VERBOSE= "      + verbose            + '\n' +                 # If ON the program will print out information.
              "DIAGNOSTICS= "  + diagnostics        + '\n' +                 # If OFF it will not print diagnostics, otherwise:
                                                                             # NOCHECKPARS
                                                                             # INSPECTINPUT
                                                                             # INSPECTSYSPARS
                                                                             # INSPECTSTAR2TILES
                                                                             # INSPECTDISKTILES
                                                                             # INSPECTYLIMITS
                                                                             # INSPECTHEATING
                                                                             # INSPECTESCAPE (requires an orbital phase)
              "STAR1= "        + star1              + '\n' +                 # ON or OFF, compact object
              "STAR2= "        + star2              + '\n' +                 # ON or OFF, donor star
              "DISK= "         + disk               + '\n' +                 # ON or OFF, accretion disk
              "DISKRIM= "      + diskrim            + '\n' +                 # ON or OFF, the disk has a raised rim
              "DISKTORUS= "    + disktorus          + '\n' +                 # ON or OFF, the disk has a raised torus at an intermediate radius
              "INNERDISK= "    + innerdisk          + '\n' +                 # ON or OFF, the disk has a small flat inner disk
              "DISKSPOTS= "    + diskspots          + '\n' +                 # ON or OFF, the disk can have up to 19 spots
              "ADC= "          + ADC                + '\n' +                 # ON or OFF, Accretion Disk Corona
              "THIRDLIGHT= "   + thirdlight         + '\n' +                 # ON or OFF, the light curve has an added third light
              "IRRADIATION= "  + irradiation        + '\n' +                 # ON or OFF, the model includes heating by irradiation
              "PHASES= "       + min_phase+" "+max_phase+" "+step_size+'\n'+ # Phase coverage and resolution of light curve.
                                                                             # Phase 0 is inferior conjunction of secodnary
              "PHASEOFFSET= "  + phaseoffset        + '\n' +                 # Offset of light curve in fractions of the orbit.
              "BANDPASS= "     + "FILTER "+ band    + '\n' +                 # FILTER or SQUARE:
                                                                             # FILTER will calculate U, B, V, R, I, J, K, etc.
                                                                             # SQUARE will calculate bounded by a min and max wavelength.
                                                                             # SQUARE minwave maxwave
              "NORMALIZE= "    + "FITDATA "+ band   + '\n' +                 # OFF, MAXVALUE, or FITDATA
                                                                             # OFF won't normalize the light curve
                                                                             # MAXVALUE maxvalue: will normalize to that value at the phase of minimum flux.
                                                                             # FITDATA filtername: Normalize by calculating the minimum variance value.
                                                                             # FITDATA SQUARE minwave maxwave: Normalize by calculating the minimum variance value.
              "PERIOD= "       + period             + '\n' +                 # Orbital period of the system in days
              "M1= "           + m1                 + '\n' +                 # Mass of the primary star in solar masses
              "MASSRATIO= "    + massratio          + '\n' +                 # Mass ratio M2 / M1
              "INCLINATION= "  + inclination        + '\n' +                 # Orbital inclination in degrees
              "STAR1LUM= "     + star1lum           + '\n' +                 # Primary Luminosity in ergs/second
              "STAR1TEMP= "    + star1temp          + '\n' +                 # Primary effective temperature in Kelvins
              "STAR2TILES= "   + star2tiles         + '\n' +                 # Target number of tiles to cover the surface of star
              "STAR2TEMP= "    + star2temp          + '\n' +                 # Secondary star mean temperature
              "STAR2ALBEDO= "  + star2albedo        + '\n' +                 # Albedo of star 2
              "DISKTILES= "    + disktiles          + '\n' +                 # Target number of tiles to cover the accretion disk
              "DISKE= "        + diske              + '\n' +                 # Eccentricity of the elliptical disk
              "DISKZETAZERO= " + diskzetazero       + '\n' +                 # Angle of periastron of the disk in degrees
              "DISKALBEDO= "   + diskalbedo+";"     + '\n' +                 # Albedo of the disk
              "MAINDISKA= "    + r_isco+" "+r_disk  + '\n' +                 # amin, amax: The minimum and maximum semi-major axis of the main disk
              "MAINDISKH= "    + disk_h+" "+disk_h_power    + '\n' +         # Height, Power: H = Height (a - a_min / a_max - a_min) ^ power
              "MAINDISKT= "    + "VISCOUS " + maindiskt     + '\n' +         # VISCOUS maindiskL: Steady-state, optically thick, viscous disk. maindiskL is luminosity.
                                                                             # Powerlaw Pow mainsidkL(K): Temperature distribution T = K*a ^pow
              "DISKEDGET= "    + edge_T+" "+edge_T_spot+" "+edge_spot_angle+" "+edge_spot_width + '\n' + # Tedge Tspot ZetaMid ZetaWidth:
                                                                             # Tedge: temperature at edge of the disk
                                                                             # ZetaMid and Zetawidth and parameters for Tspot, in degrees.
                                                                             # Extends from (Zetamid - Zetawidth/2) to (Zetamid + Zetawidth/2)
              "READDATA= "     + "FILTER "+band+ " " + File_name + '\n' +    # FILTER filtername filename
                                                                             # SQUARE minwave maxwave filename
                                                                             # Read in the file with observed light curve
              "END"]

    # Save the data to file
    LCFile = open(ParFile, "w")
    LCFile.write(output[0])

def binary_model(theta, File_binned_data, letter, shift_periastron = False):
    '''
    This function will run XRbinary given a set of input parameters.
 
    Parameters
    ----------
    X_P              : Values of parameters
    File_binned_data : Name of the file with the binned data
    letter           : Letter for the ParFileName, to abvoid overlap

    Returns
    -------
    The interpolation function
    '''

    phaseoffset_P, inclination_P, star2temp_P, maindiskt_P, massratio_P, diske_P, diskzetazero_P, disk_size_P, edge_T_P, diskh_P = theta

    # Random number generator to save the ParFile
    ParFileName = "XY" + letter + str(int(np.random.mtrand.RandomState().randint(10000000)))

    # Calculate The size of the accretion disk in units of semi-major axis
    # using the Eggelton Approximation
    q = 1.0 / float(massratio_P)
    R1_A = 0.49 * q ** (2/3.0) / (0.6 * q ** (2/3.0) + np.log(1.0 + q ** (1/3.0)))

    # Radius of the accretion disk = 0.9 of Roche Lobe
    r_disk_P = np.around(disk_size_P * R1_A, decimals = 8)

    # Calculate the temperature of the spot (Default is edge_T_spot == 1; the same as edge_T)
    spot_temperature = edge_T_P * edge_T_spot

    # If necessary, shift the definition of the disk's argument of periastron by 180 degrees.
    if shift_periastron:
        if diskzetazero_P >= 0:
            diskzetazero_mod_P = diskzetazero_P
        elif diskzetazero_P < 0:
            diskzetazero_mod_P = diskzetazero_P + 360.0
    else:
        diskzetazero_mod_P = diskzetazero_P

    # Calcualte primary mass from velocity ampltidue
    m1_P = (1 + massratio_P) ** 2 / (np.sin(inclination_P / 180 * np.pi)) ** 3 * (period * 86400 * k2_center ** 3 / (2 * np.pi * G))

    # Create Parameter File
    create_parfile(ParFileName,
                   phaseoffset     = str(phaseoffset_P), inclination     = str(inclination_P)   , star2temp       = str(star2temp_P)       , maindiskt       = str(maindiskt_P)    ,
                   massratio       = str(massratio_P)  , diske           = str(diske_P)         , diskzetazero    = str(diskzetazero_mod_P), m1              = str(m1_P)           ,
                   edge_T          = str(edge_T_P)     , edge_T_spot     = str(spot_temperature), edge_spot_angle = str(edge_spot_angle)   , edge_spot_width = str(edge_spot_width),
                   r_disk          = str(r_disk_P)     , star1lum        = str(star1lum)        , star1temp       = str(star1temp)         , min_phase       = str(min_phase)      ,
                   max_phase       = str(max_phase)    , step_size       = str(step_size)       , band            = str(band)              , period          = str(period)         ,
                   star2tiles      = str(star2tiles)   , star2albedo     = str(star2albedo)     , disktiles       = str(disktiles)         , diskalbedo      = str(diskalbedo)     ,
                   r_isco          = str(r_isco)       , disk_h          = str(diskh_P)         , disk_h_power    = str(disk_h_power)      , File_name       = File_binned_data    )

    # Run the XRbinary Code
    process = subprocess.Popen('./a.out ' + ParFileName, shell = True, stdout=subprocess.PIPE)
    process.wait()

    # Save the likelihood function that the code returns
    output_in = process.communicate()

    # Fix the format out the output, depending on Python version
    if float(platform.python_version()[0]) > 2:
        output = str(output_in[0], 'utf-8')
    else:
        output = str(output_in[0])

    # Remove Parameter File
    subprocess.call('rm ' + ParFileName, shell = True)

    try:
        # Find the phases and fluxes values in the output
        eses = [i for i, letra in enumerate(output) if letra == 'S']
        tees = [i for i, letra in enumerate(output) if letra == 'T']

        # Extract the phase out of the output
        phases_out = []
        for i in range(len(tees)):
            phases_out = np.append(phases_out, float(output[eses[i]+1:tees[i]]))

        # Extract the flux out of the output
        fluxes_out = []
        for i in range(len(tees)-1):
            fluxes_out = np.append(fluxes_out, float(output[tees[i]+1:eses[i+1]]))

        # Append the last index of the fluxes
        if len(fluxes_out) == len(phases_out) - 1:
            fluxes_out = np.append(fluxes_out, float(output[tees[-1]+1:]))

        # Modify the range of the last data point to prevent
        # the "Data point outside of range" error in interpolation
        phases_out[-1] += 0.00001
        phases_out[0]   = 0.00000

        # Do an interpolation of all the flux points.
        f = interp1d(phases_out, fluxes_out, kind = 'cubic')

        # Output function
        return f
    except:
        print("XRbinary Error")
        return - np.inf

def calc_likelyhood(Interpolation_Function, Raw_Time, Raw_Light, Raw_Error):
    '''
    Calculate the likelihood given the Interpolation_Function from 
    fitting the model, and the raw data
    '''
    Output_Flux_Data = Interpolation_Function(Raw_Time)
    weight = 1.0 / (Raw_Error ** 2)
    error  = Raw_Light - Output_Flux_Data
    Likely = -0.5*(np.sum(weight * error ** 2 + np.log(2.0 * np.pi / weight)))

    return Likely

# Log of probabilty function
def lnlike(theta, Raw_Time_a, Raw_Light_a, Raw_Error_a, do_plot = False):
    '''
    Calculates the log of the likelihood function

    Parameters
    ----------
    theta   : list of parameters input in the model
    x       : values of time, x axis (unbinned)
    y       : flux/velocity values, y axis (unbinned)
    yerr    : sigma errors in the y data (unbinned)
    do_plot : Save a plot of the output?

    Returns
    -------
    np.array: log likelihood
    '''
    phaseoffset_P, inclination_P, star2temp_P, maindiskt_P, massratio_P, diske_P, diskzetazero_P, disk_size_P, edge_T_P, diskh_P = theta

    # How long did this model take
    WD_Start = datetime.datetime.now()

    # Likelihood 
    Interpolation_Function_a = binary_model(theta, File_name, "a")

    # If specified, plot the data
    if do_plot:
        Output_Flux_Data_a = Interpolation_Function_a(Raw_Time_a)
        plt.title(
        r'$\phi = $'      + str(np.around(phaseoffset_P         , decimals = 2)) + '  ' +  
        r'$i = $'         + str(np.around(inclination_P         , decimals = 2)) + '  ' +  
        r'$T_2 = $'       + str(np.around(star2temp_P           , decimals = 2)) + '  ' + '\n' +
        r'$\log(T_D) = $' + str(np.around(np.log10(maindiskt_P) , decimals = 2)) + '  ' +  
        r'$q = $'         + str(np.around(massratio_P           , decimals = 2)) + '  ' +  
        r'$K_2 = $'       + str(np.around(k2_center             , decimals = 2)) + '  ' +  
        r'$e = $'         + str(np.around(diske_P               , decimals = 2)) + '  ' + '\n' +
        r'$\omega = $'    + str(np.around(diskzetazero_P        , decimals = 2)) + '  ' +  
        r'$D_R = $'       + str(np.around(disk_size_P           , decimals = 2)) + '  ' +  
        r'$T_E = $'       + str(np.around(edge_T_P              , decimals = 2)) + '  ' +  
        r'$H_D = $'       + str(np.around(diskh_P               , decimals = 2)) + '  ' )
        plt.errorbar(Raw_Time_a, Raw_Light_a, Raw_Error_a, color = 'g', alpha = 0.5, fmt = 'o')
        plt.plot(Raw_Time_a, Output_Flux_Data_a, color = 'k', linewidth = 1)
        plt.savefig('output_plot.jpg', bbox_inches = 'tight', dpi = 200)
        plt.clf()

    # Calculate the likelihood as XRbinary would
    try:
        Likely_a = calc_likelyhood(Interpolation_Function_a, Raw_Time_a, Raw_Light_a, Raw_Error_a)
        print("Good")
    except Exception as error_code:
        print("Interpolation Error: \n     " + str(error_code))
        return - np.inf

    print(Likely_a)
    return Likely_a

# Log prior funtion, define the priors here
def lnprior(theta):
    '''
    Calculated likelihood penalties for choice of parameters.
    Any parameter outside of the prior will be excluded
 
    Parameters
    ----------
    theta: list of input parameters in the model
    in this version:
        - prior on inclination is flat in cos(i)
        - prior on vsini is Gaussian

    Returns
    -------
    number:
    -inf if the values are outside the prior.
    '''

    phaseoffset_P, inclination_P, star2temp_P, maindiskt_P, massratio_P, diske_P, diskzetazero_P, disk_size_P, edge_T_P, diskh_P = theta
 
    # Calculate the vsini, M1, and M2
    vsini_P = 0.462 * k2_center * (massratio_P * (1 + massratio_P)**2)**(1./3)
    m1_P    = (1 + massratio_P) ** 2 / (np.sin(inclination_P / 180 * np.pi)) ** 3 * (period * 86400 * k2_center ** 3 / (2 * np.pi * G))
    m2_P    = massratio_P * m1_P

    # Calculate the radius of the accretion disk
    q        = 1.0 / float(massratio_P)
    R1_A     = 0.49 * q ** (2/3.0) / (0.6 * q ** (2/3.0) + np.log(1.0 + q ** (1/3.0)))
    r_disk_P = disk_size_P * R1_A

    if r_disk_P > diskh_P and \
       min_phaseoffset     <   phaseoffset_P   < max_phaseoffset     and \
       min_inclination     <   inclination_P   < max_inclination     and \
       min_star2temp       <    star2temp_P    < max_star2temp       and \
       min_massratio       <    massratio_P    < max_massratio       and \
       min_diske           <      diske_P      < max_diske           and \
       min_diskzetazero    <   diskzetazero_P  < max_diskzetazero    and \
       min_edge_T          <     edge_T_P      < max_edge_T          and \
       10**min_10maindiskt <    maindiskt_P    < 10**max_10maindiskt and \
       min_disk_size       <    disk_size_P    < max_disk_size       and \
       min_vsini           <      vsini_P      < max_vsini           and \
       min_diskh           <      diskh_P      < max_diskh           and \
       min_m2              <        m2_P       < max_m2                  :
        return (np.log(1.0 / (maindiskt_P * (max_10maindiskt - min_10maindiskt))) +
                np.log(np.sin(inclination_P * np.pi / 180)  / (max_inclination - min_inclination)) +
                np.log(1.0 / (max_phaseoffset - min_phaseoffset)) +
                np.log(1.0 / (max_star2temp - min_star2temp)) +
                np.log(gaussian(vsini_P, vsini_center, width_vsini)  / (max_vsini - min_vsini)) +
                np.log(1.0 / (max_vsini - min_vsini)) +
                np.log(1.0 / (max_disk_size - min_disk_size)) +
                np.log(1.0 / (max_edge_T - min_edge_T)) +
                np.log(1.0 / (max_massratio - min_massratio)) +
                np.log(1.0 / (max_diskh - min_diskh)) +
                np.log(1.0 / (max_m2 - min_m2)))
    print("Outside Prior")
    return -np.inf

# Posterior function
def lnprob(theta, Raw_Time_a, Raw_Light_a, Raw_Error_a):
    '''
    Sums the log likelihood and the log prior
 
    Parameters
    ----------
    theta: list of parameters input in the model
    x: values of time, x axis (binned)
    y: flux/velocity values, y axis (binned)
    yerr: sigma errors in the y data (binned)

    Returns
    -------
    np.array: sum of log prior and log likelihood
    '''

    # Model counter
    global itera
    itera += 1
    print( "")
    print("Iteration = " + str(itera * n_cores) + " / " + str(n_walkers * (n_steps+1)))
    print(theta)

    # Return likelihood function + Prior
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, Raw_Time_a, Raw_Light_a, Raw_Error_a)

##################################
########## CALCULATIONS ##########
##################################

##### Convert Magnitude to Luminosity if necessary #####
def convert_to_lum(Light_data, Error_data, band):
    '''
    Convert Magnitude to Luminosity if necessary
    Solar Magnitudes are taken from http://mips.as.arizona.edu/~cnaw/sun.html
    '''
    if band == 'U': M_Sun = 5.55
    if band == 'B': M_Sun = 5.45
    if band == 'V': M_Sun = 4.80
    if band == 'R': M_Sun = 4.46
    if band == 'I': M_Sun = 4.11
    if band == 'J': M_Sun = 3.67
    if band == 'H': M_Sun = 3.33
    if band == 'K': M_Sun = 3.29

    # Distance to the Sun in kpc
    d_Sun = 4.84814E-9

    # Solar Luminosity in erg/s
    L_Sun = 3.839E33

    Luminosity_data = 10 ** (0.4 * (M_Sun - Light_data)) * L_Sun * (d_star / d_Sun) ** 2
    Sigma_data      = np.abs((-0.921034 * np.exp(0.921034 * M_Sun - 0.921034 * Light_data)) * Error_data * L_Sun * (d_star / d_Sun) ** 2)

    return Luminosity_data, Sigma_data

#Luminosity_data_a, Sigma_data_a = convert_to_lum(Light_data_a, Error_data_a, band)
Luminosity_data_a, Sigma_data_a = Light_data_a, Error_data_a

##### MCMC starts here #####
# Determine the initial positions of each of the n_walkers walkers.
pos_specs = []

# Add ranges for parameters
pos_specs.append(['u'  , min_phaseoffset   , max_phaseoffset   ]) # Phase Offset
pos_specs.append(['u'  , min_inclination   , max_inclination   ]) # Inclination (Meaningless, get's replaced)
pos_specs.append(['u'  , min_star2temp     , max_star2temp     ]) # Secondary Temperature
pos_specs.append(['j'  , min_10maindiskt   , max_10maindiskt   ]) # Disk Luminosity
pos_specs.append(['u'  , min_massratio     , max_massratio     ]) # Mass ratio M2 / M1
pos_specs.append(['u'  , min_diske         , max_diske         ]) # Disk Eccentricity
pos_specs.append(['u'  , min_diskzetazero  , max_diskzetazero  ]) # Argument of Periastron
pos_specs.append(['u'  , min_disk_size     , max_disk_size     ]) # Disk Size
pos_specs.append(['u'  , min_edge_T        , max_edge_T        ]) # Edge Temperature
pos_specs.append(['u'  , min_diskh         , max_diskh         ]) # Disk Height

# Create initial walker positions
pos = create_dists(n_walkers,pos_specs)

###########
# Override inclination to make it flat in cosine space
flat_cos = np.random.random(n_walkers)
cos_rand = np.arccos(flat_cos) * 180 / np.pi

# Make sure it's in the prior range
good = np.where((cos_rand > min_inclination) & (cos_rand < max_inclination))[0]

# Keep adding values until the length is ok
cos_good = cos_rand[good]

while len(cos_good) < n_walkers:
    cos_good = np.append(cos_good, np.arccos(np.random.random(n_walkers)) * 180 / np.pi)
    good = np.where((cos_good > min_inclination) & (cos_good < max_inclination))[0]
    cos_good = cos_good[good]

# Crop to the correct size
cos_good = cos_good[0:n_walkers]

# Replace in "pos"
pos.T[1] = cos_good
###########

# Completely OVERRIDE THE INITIAL GUESS
# By Importing the one from a previus run
# pos = np.genfromtxt("Previous.txt").T

# Run the MCMC
print ("Running MCMC")
ndim = pos.shape[1]
sampler = emcee.EnsembleSampler(n_walkers, ndim, lnprob, args=(Phase_data_a, Luminosity_data_a, Sigma_data_a), threads=n_cores)

# Temporary Chain
f = open("Temporary_%s"%File_name, "w")
f.close()

# Save Incremental Steps
for result in sampler.sample(pos, iterations=n_steps):
    position = result[0]
    f = open("Temporary_%s"%File_name, "a")
    for k in range(position.shape[0]):
        np.savetxt(f,position[k],newline=" ")
        f.write("\n")
    f.close()

End = datetime.datetime.now()
print( "Model took = " + str(End - Start))

