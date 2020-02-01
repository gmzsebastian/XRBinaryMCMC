/***************************************
*
*                     FILE  STAR1.C
*
*   Functions concerned with the compact star ("star 1" or the
*   "primary star") are in this file.
*
****************************************/

#include "header.h"

double Star1TotFlux( double distance )
/*****************************************************
*
*   This function returns the integrated flux from star 1.
*   The flux has been integrated over wavelength and over 
*   the surface of the star.
*
**************************************************************/
{
  double totalflux;

  totalflux = star1.L / (FOURPI * distance * distance );
  return( totalflux );
}


double Star1Flambda( char *filter, double minlambda, double maxlambda )
/*****************************************************
*
*   This function returns the contribution of star 1 to the
*   observed spectrum at wavelength lambda.  Note that the wavelengths
*   must be in centimeters but the returned flux is in
*   ergs/sec/cm^2/Angstrom.
*
**************************************************************/
{
   double flux;

   if( strcmp( filter, "SQUARE") == 0 ) {
      flux = ( star1.L / (4.0 * star1.sigmaT4) )
              * BBSquareIntensity( star1.T, minlambda, maxlambda );
   }
   else {
      flux = ( star1.L / (4.0 * star1.sigmaT4) )
	      * BBFilterIntensity( star1.T, filter );
   }

   return( flux );
}


double MeanRocheRadius( double q )
/*************************************
*
*  This function returns the mean radius of the Roche lobe
*  in units of the separation of the two stars, e.g., <R_lobe>/a.
*  using Eggleton's (1983, ApJ, 268, 368) formula.
*
*   q = (Mass of star inside the Roche lobe) / (Mass of the other star)
*
************************************************/
{
   double x, radius;

   x = pow( q, 0.3333333);

   radius = 0.49 * x * x / ( 0.6 * x * x + log(1 + x) );

   return( radius );
}