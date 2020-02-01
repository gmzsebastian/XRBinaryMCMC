/***************************************
*
*                     FILE  DISKFLUX.C
*
*   Functions concerned with the temperature of the accretion disk
*   around the compact star are in this file.
*
****************************************/

#include "header.h"


double DiskTopT( double a, double zeta )
/***********************************************
*
*   This function returns the temperature of the top of the disk.
*   The temperature of the rim overrides the temperature of the 
*   disk if its temperature is greater than the temperature of 
*   the underlying main disk.
*
*******************************************************/
{
   double temperature, diskT, rimT, torusT, x;
  
   if( (a > maindisk.amax) || (a < maindisk.amin) ) {
      temperature = 0.0;
      return( temperature );
   }
   temperature = MainDiskT( a, zeta );

   if( strcmp( control.disktorus, "ON") == 0 ) {
      if( (a <= (disktorus.azero + 0.5 * disktorus.awidth))
               && (a >= (disktorus.azero - 0.5 * disktorus.awidth)) ) {
	 torusT = DiskTorusT( a, zeta);
         if( torusT > temperature ) {
            temperature = torusT;
         }
         return( temperature );
      }
   }
   if( strcmp( control.diskrim, "ON") == 0 ) {
      if( a >= (maindisk.amax - diskrim.awidth) ) {
         rimT = DiskRimT( a, zeta);
         if( rimT > temperature ) {
            temperature = rimT;
         }
         return( temperature );
      }
   }

   return( temperature );
}


double MainDiskT( double a, double zeta )
/**************************************
*
*   This function returns the _mean_ temperature of tiles on
*   the main disk.
*   The function returns temperature = 0 if a is not in the range
*             (amin + 0.5*da < a < (amax - 0.5*da)
*
*   The initial temperature distribution is axisymmetric and can 
*   be that of either a steady-state viscous disk or a power law 
*   of the form
*       T = Tzero * r^Tpow
*
*   The temperatures are normalized to give the correct main disk
*   luminosity, which is an input parameter.
*
*   The function then checks whether (a,zeta) falls within a disk spot
*   and then if so, calculates the revised T from
*       T = T * diskspot.spotToverT
*   Note that if there is more than one spot and the spots overlap,
*   the fraction by which the spot temperature is reduced are
*   multiplied in the overlap region.
*
*****************************************/
{
   long i;
   double x, ymin, ymax, r1, r2, T4, alpha, beta, temperature;

   /*****************************
   *
   *  If the first call to the function, calculate TConstant for
   *  future use.  TConstant is based on a disk with no spots.
   *  For a viscous disk,
   *    TConstant = 3GM dotM / (4 \pi sigma)
   *  For a power law temperature distribution
   *    TConstant = T_0^4
   *
   ****************************************/
   if( maindisk.TConstant <= 0.0 ) {
      if( strcmp(maindisk.Ttype, "VISCOUS")     == 0) {
         ymin = 1.0 / 3.0;
         ymax = 1.0 - (2.0 /3.0) * sqrt(maindisk.amin / maindisk.amax);
         x = (1.0 / maindisk.amin) * ymin
             - (1.0 / maindisk.amax) * ymax;
         maindisk.TConstant = maindisk.maindiskL / ( 2.0 * PI * SIGMA * x);	     
      }
      else if( strcmp(maindisk.Ttype, "POWERLAW") == 0) {
         beta = 4.0 * maindisk.Tpow + 2.0;
         /***********************
         *  Special exception if maindisk.Tpow == -0.5
         ************************/
         if( maindisk.Tpow == -0.5 ) {
            x = maindisk.maindiskL / (4.0 * PI * SIGMA );
            maindisk.TConstant = x / log(maindisk.amax / maindisk.amin);
         }
         else {
            x = (beta * maindisk.maindiskL) / (4.0 * PI * SIGMA );
            maindisk.TConstant = x /
                      ( pow(maindisk.amax, beta) - pow(maindisk.amin, beta));
           if(maindisk.TConstant < 0.0)
                 maindisk.TConstant = -maindisk.TConstant;
         }
      }
      else
         Quit("Unrecognized temperature distribution in function MainDiskT");
   }

   if( (a < (maindisk.amin + 0.5*maindisk.da)) 
                   || (a > (maindisk.amax - 0.5*maindisk.da)) )
      temperature = 0.0;
   else {
      if( strcmp(maindisk.Ttype, "VISCOUS")     == 0) {
         r1 = a - maindisk.da / 2.0;
         r2 = a + maindisk.da / 2.0;
         ymin = 1.0 - (2.0 / 3.0) * sqrt( maindisk.amin / r1);
         ymax = 1.0 - (2.0 / 3.0) * sqrt( maindisk.amin / r2);
         x = (1.0 / r1) * ymin - (1.0 /r2 ) * ymax; 
         T4 = (maindisk.TConstant * x) / ( r2*r2 - r1 * r1);
         temperature = pow ( T4, 0.25);
      }
      else if( strcmp(maindisk.Ttype, "POWERLAW") == 0) {
         r1 = a - maindisk.da / 2.0;
         r2 = a + maindisk.da / 2.0;
         /***********************
         *  Special exception if maindisk.Tpow == -0.5
         ************************/
         if( maindisk.Tpow == -0.5 ) {
            T4 = 2.0 * maindisk.TConstant;
            T4 = T4 * log( r2 / r1 ) / (r2*r2 - r1*r1);
         }
         else {
            beta = 4.0 * maindisk.Tpow + 2.0;
            T4 = 2.0 * maindisk.TConstant / beta;
            T4 = T4 * ( pow( r2, beta) - pow( r1, beta)) / (r2*r2 - r1*r1);
         }
         temperature = pow ( T4, 0.25);
      }
      if( strcmp( control.diskspots, "ON") == 0 ) {
         for( i = 1; i <= diskspot.nspots; i++ ) {
            if( (zeta >= diskspot.zetamin[i]) 
                             && (zeta <= diskspot.zetamax[i]) ) {
               if( (a >= diskspot.amin[i]) 
                             && (a <= diskspot.amax[i]) ) {
                  temperature *= diskspot.spotToverT[i];
               }
            }
         }
      }
   }
   return( temperature );
}


double DiskRimT( double a, double zeta )
/**************************************
*
*   This function returns the temperature of the disk rim.
*   There are currently two possibilities:
*   
*   1) DISKRIMH=  SINUSOID
*      The rim temperature is a sinusoidal function of zeta:
*         T =    0.5 * (Tmax + Tmin)
*              + 0.5 * (Tmax - Tmin) * cos( zeta - zetaTmax )
*
*   2) DISKRIMH=  POINT
*      The disk rim temperature is defined by a set of points,
*      one point per DISKRIM= line in the parameter file:
*            DISKRIM=  POINT   Zeta1   H1   T1
*            DISKRIM=  POINT   Zeta2   H2   T2    
*               .        .       .     .    .
*               .        .       .     .    .
*      The Zetas must be in increasing order and diskrim.PointZeta[1]
*      must be 0 degrees (this avoids messy computer code).
*      The temperature is linearly interpolated between the points.
*
*   The weird factor (1.0 + 1.0e-7) is protection against 
*   roundoff error at the edge of the disk.
*
*****************************************/
{
   long i;
   double temperature, zetalow, zetahigh, Tlow, Thigh, slope, ratio;

   if( zeta > TWOPI )
      Quit("zeta greater than TWOPI in DiskRimT.");
   if( zeta < 0.0 )
      Quit("zeta less than zero in DiskRimT.");

   if( a < (maindisk.amax - diskrim.awidth) ) {
      temperature = 0.0;
      return( temperature );
   }
   if( a > (1.0 + 1.0e-7) * maindisk.amax ) {
      temperature = 0.0;
      return( temperature );
   }

   if( strcmp( diskrim.type, "SINUSOID" ) == 0 ) {
      temperature =   0.5 * ( diskrim.Tmax + diskrim.Tmin )
                    + 0.5 * ( diskrim.Tmax - diskrim.Tmin )
                           * cos( zeta - diskrim.ZetaTmax );
   }
   else if( strcmp( diskrim.type, "POINT" ) == 0 ) {
      if( diskrim.points == 1 ) {
         temperature = diskrim.PointT[1];
      }
      else {
	 for( i = 1; i <= diskrim.points; i++) {
	    zetalow = diskrim.PointZeta[i];
            Tlow = diskrim.PointT[i];
            if( i < diskrim.points ) {
	       zetahigh = diskrim.PointZeta[i+1];
               Thigh = diskrim.PointT[i+1];
            }
            else {
	       zetahigh = TWOPI;
               Thigh = diskrim.PointT[1];
            }
	    if( (zeta >= zetalow) && (zeta < zetahigh) ) {
	       slope = (Thigh - Tlow) / (zetahigh - zetalow);
               temperature = Tlow + slope * (zeta - zetalow);
               break;
            }
	 }
      }
   }
   else
      Quit("Unrecognized disk rim type in DiskRimT.");

   return( temperature );
}


double DiskTorusT( double a, double zeta )
/**************************************
*
*   The temperature of the torus varies with zeta but not a.
*   There are currently two possibilities for the zeta dependence
*
*   1) SINUSOID
*      T = 0.5 * (Tmax + Tmin)
*            + 0.5* (Tmax - Tmin) * cos( zeta - zetaTmax );
*
*   2) POINT
*      The disk torus height and temperature is defined by a set of
*      points, one point per DISKTORUSPARS= line in the parameter file:
*         DISKTORUSPARS=  POINT   Zeta1   H1   T1
*         DISKTORUSPARS=  POINT   Zeta2   H2   T2    
*            .        .       .     .    .
*            .        .       .     .    .
*      The temperatures are linearly interpolated between the specified 
*      points.
*
*      The Zetas must be in increasing order and disktorus.PointZeta[1]
*      must be 0 degrees (this avoids messy computer code).
*
*****************************************/
{
   long i;
   double Tzeta, zetalow, zetahigh, Tlow, Thigh, slope, 
          x, y, temperature;

   if( zeta > TWOPI )
      Quit("zeta greater than TWOPI in DiskTorusT.");
   if( zeta < 0.0 )
      Quit("zeta less than zero in DiskTorusT.");

   if( a < (disktorus.azero - 0.5 * disktorus.awidth) ) {
      temperature = 0.0;
      return( temperature );
   } 
   if( a > (disktorus.azero + 0.5 * disktorus.awidth) ) {
      temperature = 0.0;
      return( temperature );
   }

   if( strcmp( disktorus.type, "SINUSOID" ) == 0 ) {
      temperature =    0.5 * ( disktorus.Tmax + disktorus.Tmin )
                     + 0.5 * ( disktorus.Tmax - disktorus.Tmin ) 
                                   * cos( zeta - disktorus.ZetaTmax );
   }
   else if( strcmp( disktorus.type, "POINT" ) == 0 ) {
      if( disktorus.points == 1 ) {
         temperature = disktorus.PointT[1];
      }
      else {
	 for( i = 1; i <= disktorus.points; i++) {
	    zetalow = disktorus.PointZeta[i];
            Tlow = disktorus.PointT[i];
            if( i < disktorus.points ) {
	       zetahigh = disktorus.PointZeta[i+1];
               Thigh = disktorus.PointT[i+1];
            }
            else {
	       zetahigh = TWOPI;
               Thigh = disktorus.PointT[1];
            }
	    if( (zeta >= zetalow) && (zeta < zetahigh) ) {
	       slope = (Thigh - Tlow) / (zetahigh - zetalow);
               temperature = Tlow + slope * (zeta - zetalow);
               break;
            }
	 }
      }
   }
   else
      Quit("Unrecognized disk torus type in DiskTorusT.");

   return( temperature );
}


double DiskEdgeT( double zeta )
/**************************************
*
*   This function returns the temperature of the outer
*   edge of the disk.  The temperature is set to diskedge.T.
*
*   In addition, if diskedge.Tspot is greater than diskedge.T
*   a spot is painted onto the edge of the disk between angles
*   Zetamin and Zetamax.
*   If Zetamin is greater than Zetamax the spot is assumed 
*   to run through 360 degrees.
*
*****************************************/
{
   double z1, z2, temperature;

   temperature = diskedge.T;
   if( diskedge.Tspot > diskedge.T ) {
      z1 = diskedge.ZetaMid - diskedge.ZetaWidth / 2.0;
      z2 = diskedge.ZetaMid + diskedge.ZetaWidth / 2.0;
      if( z1 < 0.0 ) {
         if( zeta >= (z1 + TWOPI) )
            temperature = diskedge.Tspot;
	 if( zeta <= z2 )
	   temperature = diskedge.Tspot;
      }
      else if( z2 > TWOPI ) {
         if( zeta < (z2 - TWOPI) )
	    temperature = diskedge.Tspot;
	 if( zeta >= z1 )
 	    temperature = diskedge.Tspot;
      }
      else if( (zeta >= z1) && (zeta <= z2) ) {
	 temperature = diskedge.Tspot;
      }
   }
   
   return( temperature );
}


double DiskL( void )
/**********************************************************
*
*   Calculate the luminosity of the disk by adding up the fluxes
*   from all its tiles.  This function should not be used until
*   after heating by irradiation has been calculated.  Note that
*   the luminosity of the disk will be different from maindiskL
*   if there are spots, edges, torii, etc. on the disk.
*   Also find the maximum and minimum temperatures of the disk
*   tiles on the top and bottom (not the edge) of the disk.
*
************************************************************/
{
   long itile;
   double luminosity;

   luminosity = 0.0;
      disk.TopTmax = 0.0;
      disk.TopTmin = 1.0e12;
   for( itile = 1; itile <= disk.Ntiles; itile++) {
      luminosity += SIGMA * TDiskT4[itile] * TDiskdS[itile];
      if( TDiska[itile] < 0.999 * maindisk.amax ) {
         if( TDiskT[itile] > disk.TopTmax )
            disk.TopTmax = TDiskT[itile];
         if( TDiskT[itile] < disk.TopTmin )
            disk.TopTmin = TDiskT[itile];
      }
   }
   return( luminosity );
}


double InnerDiskTotFlux( double distance)
/*****************************************************
*
*   This function returns the integrated flux from the inner disk.
*   The flux has been integrated over wavelength and over 
*   the surface of the disk.  The returned quantity must by
*   multiplied by the geometric projection factor 
*          mu = cos( theta ) 
*   to get the irradiating flux.  The factor is TWOPI
*   instead of FOURPI because of the mu factor.
*
**************************************************************/
{
  double totalflux;

  totalflux = innerdisk.L / (TWOPI * distance * distance );
  return( totalflux );
}


double InnerDiskFlambda( char *filter, double minlambda, double maxlambda)
/*****************************************************
*
*   This function returns the contribution of the inner disk to the
*   observed spectrum at wavelength lambda.  Note that the wavelengths
*   must be in centimeters but the returned flux is in
*   ergs/sec/cm^2/Angstrom.  The returned quantity must be 
*   multiplied by the geometric projection factor
*                   mu = cos( theta ) 
*   to get the observed quantity.
*
**************************************************************/
{
   double flux;

   if( strcmp( filter, "SQUARE") == 0 ) {
      flux = ( innerdisk.L / (2.0 * innerdisk.sigmaT4) )
              * BBSquareIntensity( innerdisk.T, minlambda, maxlambda );
   }
   else {
      flux = ( innerdisk.L / (2.0 * innerdisk.sigmaT4) )
	      * BBFilterIntensity( innerdisk.T, filter );
   }

   return( flux );
}


double ADCTotFlux( double distance)
/*****************************************************
*
*   This function returns the integrated flux from ONE of
*   the ADC points:
*   The integrated flux is just adc.L/2.0 diluted by the
*   area of the sphere around the ADC point.
*
**************************************************************/
{
  double totalflux;

  totalflux = 0.5 * adc.L / (FOURPI * distance * distance );
  return( totalflux );
}


