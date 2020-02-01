/*************************************************
*
*               FILE  UTILITY.C
*
*   This file contains miscellaneous utility functions.
*
**************************************************/

#include "header.h"

double AngleDistance( double theta1, double phi1,
                      double theta2, double phi2)
/*****************************************************
*
*   Calculates the angular distance between two directions, where
*   the directions are given by their coordinates (theta,phi) in
*   the spherical polar coordinate system.  All angles in radians.
*
************************************************************/
{
   double cosa, a;

   cosa = cos(theta1) * cos(theta2) 
             + sin(theta1) * sin(theta2) * cos(phi2 - phi1);
   a = acos( cosa );

   return( a );
}


double CartDotProd( struct CartVector A, struct CartVector B)
/********************************************************
*
*   Calculates the dot product of two vectors, both of which
*   are in Cartesian coordinates.
*
*******************************************************/
{
   double prod;

   prod = A.x * B.x + A.y * B.y + A.z * B.z;

   return( prod );
}


struct SphereVector Cart2Sphere( struct CartVector Acart, 
                                         double theta, double phi )
/************************************
*
*   This converts a vector from Cartesian coordinates to 
*   spherical polar coordinates.  More specifically, it calculates
*   the components of the vector in spherical polar coordinates
*   given its components in Cartesian coordinates.
*
*****************************************/
{
   struct SphereVector Asphere;
   double sint, cost, sinp, cosp, m11, m12, m13, m21, m22, m23, 
          m31, m32, m33;

   sint = sin( theta );
   cost = cos( theta );
   sinp = sin( phi );
   cosp = cos( phi );

   m11 =   sint * cosp;
   m21 =   cost * cosp;
   m31 = - sinp;

   m12 =   sint * sinp;
   m22 =   cost * sinp;
   m32 =   cosp;

   m13 =   cost;
   m23 = - sint;
   m33 =   0.0;

   Asphere.r     = m11 * Acart.x + m12 * Acart.y + m13 * Acart.z;
   Asphere.theta = m21 * Acart.x + m22 * Acart.y + m23 * Acart.z;
   Asphere.phi   = m31 * Acart.x + m32 * Acart.y + m33 * Acart.z;

   return( Asphere );
}


struct CartVector Sphere2Cart( struct SphereVector Asphere, 
                                          double theta, double phi )
/************************************
*
*   This converts a vector from spherical polar coordinates to 
*   Cartesian coordinates.  More specifically, it calculates
*   the components of the vector in Cartesian coordinates
*   given its components in Spherical polar coordinates.
*
*****************************************/
{
   struct CartVector Acart;
   double sint, cost, sinp, cosp, m11, m12, m13, m21, m22, m23, 
          m31, m32, m33;

   sint = sin( theta );
   cost = cos( theta );
   sinp = sin( phi );
   cosp = cos( phi );

   m11 =   sint * cosp;
   m21 =   sint * sinp;
   m31 =   cost;

   m12 =   cost * cosp;
   m22 =   cost * sinp;
   m32 = - sint;

   m13 = - sinp;
   m23 =   cosp;
   m33 =   0.0;

   Acart.x = m11 * Asphere.r + m12 * Asphere.theta + m13 * Asphere.phi;
   Acart.y = m21 * Asphere.r + m22 * Asphere.theta + m23 * Asphere.phi;
   Acart.z = m31 * Asphere.r + m32 * Asphere.theta + m33 * Asphere.phi;

   return( Acart );
}


struct CartVector Cyl2Cart( struct CylVector Asphere, double zeta )
/************************************
*
*   This converts a vector from cylindrical coordinates to 
*   Cartesian coordinates.  More specifically, it calculates
*   the components of the vector in Cartesian coordinates
*   given its components in cylindrical coordinates.
*
*****************************************/
{
   struct CartVector Acart;
   double sinzeta, coszeta, m11, m12, m13, m21, m22, m23, m31, m32, m33;

   sinzeta = sin( zeta );
   coszeta = cos( zeta );

   m11 =   sinzeta;
   m21 =   0.0;
   m31 =   coszeta;

   m12 =   coszeta;
   m22 =   0.0;
   m32 = - sinzeta;

   m13 =   0.0;
   m23 =   1;
   m33 =   0.0;

   Acart.x = m11 * Asphere.rho + m12 * Asphere.zeta + m13 * Asphere.h;
   Acart.y = m21 * Asphere.rho + m22 * Asphere.zeta + m23 * Asphere.h;
   Acart.z = m31 * Asphere.rho + m32 * Asphere.zeta + m33 * Asphere.h;

   return( Acart );
}


double Planck( char * mode, double temperature, double lambdanu )
/********************
*
*  Returns the specific intensity emitted by the surface of a black body:
*   
*        Fnu     = ( 2*h*nu^3 / c^2)      / ( exp[ h*nu / k*T ] - 1 )
*
*        Flambda = ( 2*h*c^2 / lambda^5 ) / ( exp[ h*c / lambda*k*T ] -1 )
*
*  Input data:
*     mode         "NU"     returns Fnu     in units of erg/sec/cm^2/Hz
*                  "LAMBDA" returns Flambda in units of erg/sec/cm^2/cm
*     temperature  in degrees Kelvin
*     lambdanu     wavelength in cm if mode=LAMBDA
*                  frequency  in Hz if mode=NU
*
*  Note that the Planck function is normalized such that 
*     (integral over frequency) = ( sigma / pi ) * T**4
*  Thus, it is the monochromatic specific intensity per unit area.
*
*********************/
{
   double h, c, k, nu, lambda, x, y, Fnu, Flambda;

   h = 6.62608e-27;
   c=  2.99792e+10; 
   k = 1.38066e-16; 
   if( temperature <= 1.0 ) {
      Quit("Temperature out of range in function Planck.");
   }
   if( strcmp( mode, "NU" ) == 0 ) {
      nu = lambdanu;
      x = ( 2.0 * h * nu * nu * nu ) / ( c * c );
      y = ( h * nu ) / ( k * temperature ) ;
      Fnu = x / ( exp( y ) - 1.0 );
      return( Fnu );
   }
   else if( strcmp( mode, "LAMBDA") == 0 ) {
      lambda = lambdanu;
      x = ( 2.0 * h * c * c ) / pow( lambda, 5.0 );
      y = ( h * c ) / ( lambda * k * temperature );
      Flambda = x / ( exp( y ) - 1.0 );
      return( Flambda );
   }
   else
      Quit("Unrecognized mode in function Planck().");
      
   return(-1.0);  /*  Dummy statement to quiet a stupid compiler.  */
}


double BBSquareIntensity( double T, double lowlambda, double highlambda)
/*************************************************
*
*   This function returns the mean specific intensity emitted
*   by a black body with temperature T in a square bandpass from
*   lowlambda to highlambda.  Specifically, the function returns
*       ( \int_lowlambda^highlambda B(lambda) d lambda ) / deltalambda
*   where B(lambda) is in specific intensity form.  It uses
*   the precalculated table Zzeta.dat and does a logarithmic
*   interpolation in the table.
*
*   The wavelengths must be in centimeters but the intensity
*   is in ergs/cm^2/sec/Angstrom.
*   
*************************************************************/
{
   long nlow, nhigh;
   double h, k, c, x, Zeta1, logZeta1, Zeta2, logZeta2,
          ZetaLow, logZetaLow, ZetaHigh, logZetaHigh, 
          logZBBzetaHigh, logZBBzetaLow, slope, ZBBzeta1, logZBBzeta1,
          ZBBzeta2, logZBBzeta2, meanI;

   c = 2.99792e+10;
   h = 6.626075e-27;
   k = 1.38065e-16;

   if( lowlambda >= highlambda )
      Quit("lowlambda must be lt highlambda in BBSquareIntensity().");
   if( (lowlambda == 0) || (highlambda == 0) )
      Quit("Both wavelengths must be gt zero in BBSquareIntensity().");

   Zeta2 = ( h * c ) / (  lowlambda * k * T );
   Zeta1 = ( h * c ) / ( highlambda * k * T );
   if( (Zeta1 < deltaBBzeta) || (Zeta2 < deltaBBzeta) ) {
      Quit("lambda x T too high in BBSquareIntensity().");
   }

   if( Zeta1 >= BBzetamax ) {
      meanI = 0.0;
      return( meanI );
   }

   else {
      logZeta1 = log( Zeta1 );
      nlow = Zeta1 / deltaBBzeta;
      ZetaLow = nlow * deltaBBzeta;
      logZetaLow = log( ZetaLow );
      nhigh = nlow + 1;
      logZetaHigh = log( ZetaLow + deltaBBzeta );
      logZBBzetaLow  = log( ZBBzeta[nlow]  );
      logZBBzetaHigh = log( ZBBzeta[nhigh] );
      slope = (logZBBzetaHigh - logZBBzetaLow) / (logZetaHigh - logZetaLow);
      logZBBzeta1 = logZBBzetaLow + (logZeta1 - logZetaLow) * slope;
      ZBBzeta1 = exp( logZBBzeta1 ); 
   }

   if( Zeta2 >= BBzetamax ) {
      ZBBzeta2 = ZBBzeta[maxBBzetaindex];
   }
   else {
      logZeta2 = log( Zeta2 );
      nlow = Zeta2 / deltaBBzeta;
      ZetaLow = nlow * deltaBBzeta;
      logZetaLow = log( ZetaLow );
      nhigh = nlow + 1;
      logZetaHigh = log( ZetaLow + deltaBBzeta );
      logZBBzetaLow  = log( ZBBzeta[nlow]  );
      logZBBzetaHigh = log( ZBBzeta[nhigh] );
      slope = (logZBBzetaHigh - logZBBzetaLow) / (logZetaHigh - logZetaLow);
      logZBBzeta2 = logZBBzetaLow + (logZeta2 - logZetaLow) * slope;
      ZBBzeta2 = exp( logZBBzeta2 );
   }

   meanI = ( T*T*T*T ) * (ZBBzeta2 - ZBBzeta1) / (highlambda - lowlambda);
   meanI *= 1.0e-08;

   return( meanI );
}


double BBFilterIntensity( double T, char *filter)
/*************************************************
*
*   This function returns the mean specific intensity emitted
*   by a black body with temperature T observed through a filter.
*   
*************************************************************/
{
   long i, findex, minTindex, maxTindex;
   double slope, intensity;

   /*********************************************
   *
   *   First identify the filter index corresponding to the
   *   filter name.
   *
   *********************************************************/
   findex = -1;
   for( i = 0; i <= maxIBBfilterindex; i++) {
      if( strcmp( IBBfilterName[i], filter) == 0 ) {
         findex = i;
         break;
      }
   }
   if( findex == -1 )
      Quit("Unknown filter name in BBFilterIntensity.");

   if( (T < IBBT[0]) || (T > IBBT[maxIBBTindex]) )
      Quit("T out of range in BBFilterIntensity.");
   if( T == IBBT[maxIBBTindex] ) {
      intensity = IBBtable[maxIBBTindex][findex];
      return( intensity );
   }
   minTindex = (T - IBBTmin) / IBBdeltaT;
   maxTindex = minTindex + 1;
   slope = (IBBtable[maxTindex][findex] - IBBtable[minTindex][findex]) 
                   / IBBdeltaT;
   intensity = slope * (T - IBBT[minTindex]) + IBBtable[minTindex][findex];

   return( intensity );
}
