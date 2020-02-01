import corner
import os
import numpy as np
import matplotlib.pyplot as plt
import corner
from matplotlib import rcParams
from matplotlib import gridspec
import glob
rcParams["font.size"]  = 13

def plot_file(color, variable, gs, plot_corner = False, period = 1.9968, k2_center = 126.0, n_walkers = 40, File_name = 'Temporary_lc_gkper_mod.txt'):
    '''
    Plot a trace plot of the set of walkers output from Model_XRbinary.py

    Parameters
    ----------
    color       : Color of the plot
    variable    : variable to plot
    gs          : matplotlib grid object
    period      : Orbital period in days
    k2_center   : Semi-major velocity amplitude
    n_walkers   : Number of walkers
    plot_corner : Save a corner plot?
    File_name   : Name of file with output walkers

    Returns
    -------
    The interpolation function

    '''
    print(variable)

    # Import separate parameters for MCMC calculation
    phaseoffset, inclination, star2temp, maindiskt, massratio, diske, diskzetazero, disk_size, edge_T, diskh = np.genfromtxt(File_name, unpack = True)

    # Calculate the vsini, M1, and M2
    G     = 1.327E11 # in km^3 / (Solar masses * seconds ^ 2)
    vsini = 0.462 * k2_center * (massratio * (1 + massratio)**2)**(1./3)
    m1    = (1 + massratio) ** 2 / (np.sin(inclination / 180 * np.pi)) ** 3 * (period * 86400 * k2_center ** 3 / (2 * np.pi * G))
    m2    = massratio * m1

    # Get the samples into the right shape
    samples_mcmc = np.array([phaseoffset, inclination, star2temp, np.log10(maindiskt), massratio, diske, diskzetazero, disk_size, edge_T, diskh, vsini, m1, m2])
    samples_T    = np.transpose(samples_mcmc)
    samples_F    = samples_T[int(0.4*len(samples_T)):]

    # Get best parameters
    phaseoffset_mcmc, inclination_mcmc, star2temp_mcmc, maindiskt_mcmc, massratio_mcmc, diske_mcmc, diskzetazero_mcmc, disk_size_mcmc, edge_T_mcmc, diskh_mcmc, vsini_mcmc, m1_mcmc, m2_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples_F, [15.87, 50, 84.13], axis=0)))

    # Prior range for each parameter, assuming flat priors
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
    min_m1            , max_m1            =    0.0, 3.00

    n_dim            = len(samples_mcmc)
    n_steps          = int(len(np.array(samples_mcmc).T) / n_walkers)
    parameters_shape = np.array(samples_mcmc.flatten()).reshape(n_walkers, n_steps, n_dim)

    def rund(x, n):
        return np.around(x, decimals = n)

    def plot_trace(index, mino, maxo, best, label, title):
        print(best[0])
        parameters_shape = index.reshape(n_steps, n_walkers)
        Averageline = np.nanmean(parameters_shape, axis = 1)

        if len(np.where(np.isnan(index))[0]) > 0:
            for i in range(len(parameters_shape.T)):
                if np.nansum(parameters_shape.T[i]) == 0.0:
                    print('%s Averaged'%i)
                    parameters_shape.T[i] = Averageline
                else:
                    for j in range(len(parameters_shape)):
                        if np.isfinite(parameters_shape.T[i][j]):
                            a = 1
                        else:
                            index = j + 1
                            while np.isnan(parameters_shape.T[i][j]):
                                if index >= len(parameters_shape.T[i]):
                                    index -= 1
                                    while np.isnan(parameters_shape.T[i][j]):
                                        parameters_shape.T[i][j] = parameters_shape.T[i][index]
                                        index -= 1
                                else:
                                    parameters_shape.T[i][j] = parameters_shape.T[i][index]
                                    index += 1

        average = np.average(parameters_shape, axis = 0)

        plt.subplots_adjust(wspace=0)
        ax0 = plt.subplot(gs[0])
        ax0.axhline(best[0], color=color, lw = 2.0, linestyle = '--', alpha = 0.75)
        plt.title(r"$%s^{+%s}_{-%s}$"%(rund(best[0], 5), rund(best[1], 5), rund(best[2], 5)))
        ax0.plot(parameters_shape, '-', color=color, alpha = 0.2, lw = 0.25)
        plt.ylabel(label)
        plt.xlim(0, n_steps - 1)
        plt.ylim(mino, maxo)
        plt.xlabel("Step")

        ax1 = plt.subplot(gs[1])
        plt.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False, labeltop=False, labelright=False, labelbottom=False)
        plt.ylim(mino, maxo)
        ax1.hist(np.ndarray.flatten(parameters_shape[int(len(parameters_shape)/2):,:]), bins='auto', orientation="horizontal", color = color, alpha = 0.3)
        ax1.axhline(best[0], color=color, lw = 2.0, linestyle = '--', alpha = 0.75)

    # Plot the corner plot or trace plots?
    if plot_corner :    
        figure = corner.corner(samples_F, labels=[r'$\phi$', r'$i$', r'$T_2$', r'$L_D$', r'$q$', r'$e_D$', r'$\omega_D$', r'$R_D$', r'$T_E$', r'$H_D$', r'$vsin(i)$', r'$M_1$', r'$M_2$'], truths=[phaseoffset_mcmc[0], inclination_mcmc[0], star2temp_mcmc[0], maindiskt_mcmc[0], massratio_mcmc[0], diske_mcmc[0], diskzetazero_mcmc[0], disk_size_mcmc[0], edge_T_mcmc[0], diskh_mcmc[0], vsini_mcmc[0], m1_mcmc[0], m2_mcmc[0]], show_titles=True, quantiles=[0.1587,0.50,0.8413], title_kwargs={"fontsize": 14}, use_math_text=True, smooth=1, label_fmt='.1f')
        figure.savefig("Triangle.pdf")
        plt.clf(); plt.close('all')
    else:
        if variable == "Phi"                 : plot_trace(phaseoffset         ,  min_phaseoffset     , max_phaseoffset     , phaseoffset_mcmc         , r'$\phi$ [Phase]', "Phi"                )
        if variable == "Inc"                 : plot_trace(inclination         ,  min_inclination     , max_inclination     , inclination_mcmc         , r'$\i$ [deg]'    , "Inc"                )
        if variable == "SecondTemp"          : plot_trace(star2temp           ,  min_star2temp       , max_star2temp       , star2temp_mcmc           , r'$T_2$ [K]'     , "SecondTemp"         )
        if variable == "DiskLum"             : plot_trace(np.log10(maindiskt) ,  min_10maindiskt     , max_10maindiskt     , maindiskt_mcmc           , r'$L_D$ [erg/s]' , "DiskLum"            )
        if variable == "Mass_ratio"          : plot_trace(massratio           ,  min_massratio       , max_massratio       , massratio_mcmc           , r"$q$"           , "Mass_ratio"         )
        if variable == "Disk_Eccentricity"   : plot_trace(diske               ,  min_diske           , max_diske           , diske_mcmc               , r'$e_D$'         , "Disk_Eccentricity"  )
        if variable == "Disk_Periastron"     : plot_trace(diskzetazero        ,  min_diskzetazero    , max_diskzetazero    , diskzetazero_mcmc        , r'$\omega_D$'    , "Disk_Periastron"    )
        if variable == "Disk_Size"           : plot_trace(disk_size           ,  min_disk_size       , max_disk_size       , disk_size_mcmc           , r"$R_D$"         , "Disk_Size"          )
        if variable == "Edge_Temperature"    : plot_trace(edge_T              ,  min_edge_T          , max_edge_T          , edge_T_mcmc              , r"$T_E$"         , "Edge_Temperature"   )
        if variable == "Primary_Mass"        : plot_trace(m1                  ,  min_m1              , max_m1              , m1_mcmc                  , r"$M_1$"         , "Primary_Mass"       )
        if variable == "Secondary_Mass"      : plot_trace(m2                  ,  min_m2              , max_m2              , m2_mcmc                  , r"$M_2$"         , "Secondary_Mass"     )
        if variable == "Rotational Velocity" : plot_trace(vsini               ,  min_vsini           , max_vsini           , vsini_mcmc               , r'$vsin(i)$'     , "Rotational Velocity")
        if variable == "Disk Height"         : plot_trace(diskh               ,  min_diskh           , max_diskh           , diskh_mcmc               , r'$H_D$'         , "Disk Height"        )

def plot_figure(variable):
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1]) 
    plot_file('g', variable, gs)
    plt.savefig("Trace_plots/%s_All_Trace.jpg"%(variable), bbox_inches = 'tight', dpi = 200)
    plt.clf()

os.system('mkdir Trace_plots')
plot_figure('Phi')
plot_figure('Inc')
plot_figure('SecondTemp')
plot_figure('DiskLum')
plot_figure('Mass_ratio')
plot_figure('Disk_Eccentricity')
plot_figure('Disk_Periastron')
plot_figure('Disk_Size')
plot_figure('Edge_Temperature')
plot_figure('Primary_Mass')
plot_figure('Secondary_Mass')
plot_figure('Rotational Velocity')
plot_figure('Disk Height')

plot_file('', '', '', plot_corner = True)
