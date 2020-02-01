/***************************************
*
*                     FILE  STAR2.C
*
*   All functions concerned with the secondary star (the lobe-filling
*   star or "star 2") of the binary system are in this file.
*
****************************************/

#include "header.h"

void MakeStar2Tiles( void )
/************************************************************
*
*   This function distributes the tiles over the surface of the
*   secondary star and calculates various properties of the tiles.
*
***********************************************************/
{
   long i, j, ntheta, nphi, tilenumber, ring, band;
   double targetomega, theta1, theta2, ringarea, dtheta, dphi, 
          distance;
   double totalS, deltaS, totalgS, fourbeta, inverse4b, gratio, x;

   if( strcmp( verbose, "ONX") == 0 )
     printf(" Begin making star 2 tiles.\n");

   /****************************************************
   *
   *   First calculate the various quantities related to the
   *   geometry of the tiles on star 2.
   *   Calculating (r, theta, phi) for the center of each tile
   *   and count the true number of tiles.  For each file
   *   calculate:
   *      T2r T2theta,  T2phi,  T2dircos, 
   *      T2x, T2y, T2z
   *      T2gradV,  T2normSphere, T2g
   *      T2dS
   *      
   *   Note that the two tiles covering the two poles of the
   *   coordinate system are handled specially; and especially
   *   note the value of T2r assigned to the tile covering the
   *   pole at T2theta = PI.
   *
   *   The tiles are numbered from
   *       tile 1          = tile at the pole T2theta = 0
   *       tile tilenumber = tile at the pole T2theta = PI
   *
   **********************************************************/
   targetomega = 4.0 * PI / star2.targetNtiles;
   ntheta = 0.5 + PI / sqrt( targetomega );
   dtheta = PI / ntheta;
   T2theta[1] = 0.0;
   T2phi[1] = 0.0;
   theta1 = 0.25 * dtheta;
   T2r[1] = findR( syspars.VL1, theta1, T2phi[1] );
   theta2 = 0.5 * dtheta;
   T2gradV[1] = gradV( T2r[1], T2theta[1], T2phi[1] );
   T2normSphere[1] = Normal( T2gradV[1] );
   T2dS[1] = 2.0 * PI * (T2r[1] * dtheta/2.0) * (T2r[1] * dtheta/2.0);
   tilenumber = 1;

   for( ring = 2; ; ring++) {
      theta1 = (ring - 1.5) * dtheta;
      theta2 = theta1 + dtheta;
      if( theta2 > PI ) 
         break;
      ringarea = TWOPI * ( cos(theta1) - cos(theta2) );
      nphi = 0.5 + ringarea / targetomega;
      dphi = TWOPI / nphi;
      for( i = 1; i <= nphi; i++) {
         tilenumber += 1;
         if( tilenumber > (MAX2TILES - 2) )
            Quit("Actual number of tiles is too large.");
         T2theta[tilenumber] = 0.5 * ( theta1 + theta2 );
         T2phi[tilenumber] = (i - 0.5) * dphi;
         T2r[tilenumber] = findR( syspars.VL1, T2theta[tilenumber], 
                                  T2phi[tilenumber]);
         T2gradV[tilenumber] = gradV( T2r[tilenumber], T2theta[tilenumber], 
                                      T2phi[tilenumber] );
         T2normSphere[tilenumber] = Normal( T2gradV[tilenumber] );
         T2dS[tilenumber] = TileArea( T2r[tilenumber], T2theta[tilenumber],
                                      T2normSphere[tilenumber], dtheta, dphi );
      }
   }

   tilenumber += 1;
   T2theta[tilenumber] = PI;
   T2phi[tilenumber] = 0.0;
   T2r[tilenumber] = findR( syspars.VL1, T2theta[tilenumber], 
                            T2phi[tilenumber]);
   theta1 = PI - 0.5 * dtheta;
   T2gradV[tilenumber] = gradV( T2r[tilenumber], T2theta[tilenumber], 
                                T2phi[tilenumber] );
   T2normSphere[tilenumber] = Normal( T2gradV[tilenumber] );
   T2dS[tilenumber] = 2.0 * PI * (T2r[tilenumber] * dtheta/2.0)
                               * (T2r[tilenumber] * dtheta/2.0);

   if( abs(tilenumber - star2.targetNtiles) > (0.01 * star2.targetNtiles) )
      Quit("Actual number of tiles is much different from the target number.");
   star2.Ntiles = tilenumber;

   for( i = 1; i <= star2.Ntiles; i++ ) {
      T2x[i] = T2r[i] * sin( T2theta[i] ) * cos( T2phi[i] );
      T2y[i] = T2r[i] * sin( T2theta[i] ) * sin( T2phi[i] );
      T2z[i] = T2r[i] * cos( T2theta[i] );
      T2normCart[i] = Sphere2Cart( T2normSphere[i], T2theta[i], T2phi[i]);
      T2g[i] = - sqrt(   T2gradV[i].r     * T2gradV[i].r 
                       + T2gradV[i].theta * T2gradV[i].theta
                       + T2gradV[i].phi   * T2gradV[i].phi );
      T2logg[i] = log10( fabs(T2g[i]) );
   }

   star2.volume = 0.0;
   for( i = 1; i <= star2.Ntiles; i++) {
     deltaS = T2dS[i] * T2normSphere[i].r;
     star2.volume +=  T2r[i] * deltaS / 3.0;
   }

   x =  3.0 * star2.volume / ( 4.0 * PI);
   star2.meanr = pow( x, 0.333333333);
   star2.frontradius = syspars.rL1;
   star2.poleradius = findR( syspars.VL1, HALFPI, HALFPI);
   star2.sideradius = findR( syspars.VL1, HALFPI,    0.0);
   star2.backradius = findR( syspars.VL1,     PI,    0.0);

   /*************************************************************
   *
   *   Now calculate the various quantities related to the flux
   *   from the tiles. First calculate
   *       1) star2.beta and star2.meang;
   *   and then for each tile calculate
   *       2) T2T[] from the gravity darkening law;
   *   and finally
   *       3) attach a specific intensity to each tile for
   *          each bandpass.
   *
   *   The meanings of the specific intensity depends on the
   *   bandpass, temperature, and spectrum type:
   *
   *      square bandpass   ==> adopt a black body energy distribution.
   *                               No mu dependence, so T2I[] is the
   *                               mean specific intensity in the bandpass.
   *                               This is a very bad approximation to
   *                               the actual fluxes through a square bandpass.
   *
   *   In the following Tmin and Tmax are the minimum and maximum 
   *   temperatures in IperpTable.dat (3500 K and 8000 K as of Aug 2005).
   *
   *      filter and ( (T > Tmin) && (T < Tmax) ) 
   *                        ==> adopt a stellar atmosphere energy
   *                               distribution. T2I[] is the
   *                               perpendicular specific intensity.
   *                               Program will use a limb darkening law
   *                               to calculate the observed specific
   *                               intensity.
   *      filter & ( (T < Tmin) || (T > Tmax) )
   *                        ==> adopt a black body energy distrubution.
   *                               No mu dependence, so T2I[] is the
   *                               mean specific intensity through the
   *                               filter.
   *
   *******************************************************************/
   star2.beta = GetGDbeta();
   fourbeta = 4.0 * star2.beta;
   if( (fourbeta == 0.0) || (star2.meanT <500.0) ) {
      for( i = 1; i <= star2.Ntiles; i++) {
         T2T[i] = star2.meanT;
      }
   }
   else {
      inverse4b = 1.0 / fourbeta;
      totalS = 0.0;
      totalgS = 0.0;
      for( i = 1; i <= star2.Ntiles; i++) {
         totalS += T2dS[i];
         x = fabs( T2g[i] );
         totalgS += pow( x, fourbeta ) * T2dS[i];
      }
      x = totalgS / totalS;
      star2.meang = -1.0 * pow( x, inverse4b );
      star2.logg  = log10( fabs(star2.meang) );
      for( i = 1; i <= star2.Ntiles; i++) {
         gratio = fabs( T2g[i] / star2.meang );
         T2T[i] = star2.meanT * pow( gratio, star2.beta );
      }
   }

   /*  Note that overlapping spots multiply their temperature drops. */

   if( strcmp( control.star2spots, "ON") == 0 ) {
      for( i = 1; i <= star2spot.nspots; i++) {
         for( j = 1; j <= star2.Ntiles; j++) {
            distance = AngleDistance( star2spot.theta[i], star2spot.phi[i],
                                      T2theta[j], T2phi[j] );
            if( distance <= star2spot.radius[i] ) {
               T2T[j] = star2spot.SpotToverStarT[i] * T2T[j];
            }
         }
      }
   }

   for( band = 1; band <= orbit.nbands; band++) {
      if( strcmp( orbit.filter[band], "SQUARE") == 0 ) {
         for( i = 1; i <= star2.Ntiles; i++) {
	    T2I[band][i] = BBSquareIntensity( T2T[i], orbit.minlambda[band], 
                                                orbit.maxlambda[band]);
         }
      }
      else {
         for( i = 1; i <= star2.Ntiles; i++) {
	    if( (T2T[i] > IperpT[maxIperpTindex]) || (T2T[i] < IperpT[0]) ) {
	       T2I[band][i] = BBFilterIntensity( T2T[i], orbit.filter[band]);
            }
            else {
               T2I[band][i] = GetIperp( T2T[i], T2logg[i], orbit.filter[band]);
            }
         }
      }
   }

   if( strcmp( control.diagnostics, "INSPECTSTAR2TILES") == 0 ) {
      InspectStar2Tiles();
      Quit("Quit after INSPECTSTAR2TILES.");
   }

   return;
}

double V( double r, double theta, double phi)
/************************************
* 
*  This function calculates the potential of the zero velocity surface
*  at the point (r, theta, phi).
*
************************************/
{
   double l, n, r1, r2, rho, potential;

   l = sin(theta) * cos(phi);
   n = cos(theta);
   r1 = sqrt( syspars.a * syspars.a  +  r * r  - 2.0 * syspars.a  * r * n );
   r2 = r;
   if( (r1 <= 0.0) || (r2 <= 0.0) )
      Quit("Failure in function V.");
   rho = sqrt( (r*l)*(r*l) + (r*n - syspars.zcm)*(r*n - syspars.zcm) );
   potential =   syspars.a / r1 
               + syspars.q * syspars.a / r2 
               + 0.5 * (1 + syspars.q) * (rho / syspars.a) * (rho / syspars.a);
   potential = -(G * syspars.M1 / syspars.a) * potential;
   return( potential );
}


struct SphereVector gradV( double r, double theta, double phi)
/***********************************
*
*   This function calculates the gradient of the zero velocity potential
*   at the point (r, theta, phi).  The gradient is returned as a
*   vector in spherical polar coordinates.
*
***********************************/
{
   double epsilon, deltar, deltatheta, deltaphi;
   double dVdtheta, dVdphi, gradVr, gradVt, gradVp;
   struct SphereVector grad;

   epsilon    = 1.0e-7;
   deltar     = epsilon * syspars.a;
   deltatheta = epsilon * PI;
   deltaphi   = epsilon * TWOPI;

   gradVr = (   V( r + deltar, theta, phi) 
            - V( r - deltar, theta, phi) ) / (2.0 * deltar);
   if((theta <= deltatheta) || (theta >= (PI - deltatheta))){
      gradVt = 0.0;
      gradVp   = 0.0;
   }
   else{
      dVdtheta = (   V( r, theta + deltatheta, phi) 
                   - V( r, theta - deltatheta, phi) ) / (2.0 * deltatheta);
      gradVt   = ( 1.0 / r ) * dVdtheta;
      dVdphi   = (   V( r, theta, phi + deltaphi)
                   - V( r, theta, phi - deltaphi) )   / (2.0 * deltaphi) ;
      gradVp   = ( 1.0 / (r * sin(theta) ) ) * dVdphi;
   }
   grad.r = gradVr;
   grad.theta = gradVt;
   grad.phi = gradVp;

   return( grad );
}


struct SphereVector Normal( struct SphereVector delV )
/*********************************
*
*   This function calculates the normal vector by taking
*      normal = delV / |delV|
*   where delV is the gradient vector.  The vectors are
*   assumed to be expressed in spherical polar coordinates.
*
*********************************/
{
   struct SphereVector norm;
   double denom;

   denom = sqrt( delV.r*delV.r + delV.theta*delV.theta + delV.phi*delV.phi );
   if( denom > 0.0 ) {
      norm.r     = delV.r     / denom;
      norm.theta = delV.theta / denom;
      norm.phi   = delV.phi   / denom;
   }
   else {
      norm.r     = 1.0;
      norm.theta = 0.0;
      norm.phi   = 0.0;
   }
   
   return( norm );
}


double FindL1( void )
/*********************************
* 
*   This function finds the position of the inner Lagrangian point.
*
********************************/
{
   long iteration;
   double epsilon, epsilonr, r, rOld, slope, deltar, x;
   struct SphereVector delVplus, delVminus;
  
   epsilon = 1.0e-7;
   epsilonr = epsilon * syspars.a;
   rOld = 0.5 * syspars.a;
   for( iteration = 1; ; iteration++){
      delVplus  = gradV( rOld + epsilonr, 0.0, 0.0);
      delVminus = gradV( rOld - epsilonr, 0.0, 0.0);
      slope = (delVplus.r - delVminus.r) / (2.0 * epsilonr);
      deltar = 0.5 * (delVplus.r + delVminus.r) / slope;
      r = rOld - deltar;
      if( (r <= 0.0) || (r >= syspars.a) )
         Quit("FindL1 failed.");
      if(iteration >= 500)
         Quit("Too many iterations in function FindL1.");
      x = fabs( (rOld - r) / rOld );
      if ( x < epsilon )
         break;
      rOld = r;
   }
   return( r );
}


double findR( double Vtarget, double theta, double phi)
/*********************************************
*
*   This function finds the radius at which the zero velocity potential
*   equals targetV in the direction (theta, phi).  The radius is 
*   given as a fraction of a.
*   NOTE:  This function only works if the target potential is less than the
*   potential at the Roche lobe.
*
*********************************************/
{
   long iteration;
   double x, r, deltar, epsilonr, slope, Vminus, Vzero, Vplus;

   if( Vtarget > syspars.VL1 ) 
      Quit("Attempted to find a point outside the Roche lobe in findR).");

   epsilonr = 1.0e-7 * syspars.a;
   r = 0.8 * syspars.rL1;
   for( iteration = 1; ; iteration++) {
      if( iteration > 100 )
         Quit("Too many iterations in findR.");
      Vplus  = V( r + epsilonr, theta, phi);
      Vminus = V( r - epsilonr, theta, phi);
      Vzero  = V( r,           theta, phi);
      slope = (Vplus - Vminus) / (2.0 * epsilonr);
      if(slope == 0.0)
         slope = 100.0 * Vzero / syspars.a;
      deltar = (Vtarget - Vzero) / slope;
      if( fabs(deltar) > (0.05 * r) ) 
         deltar = (0.05 * r) * (deltar / fabs(deltar));
      r = r + deltar; 
      if( r >= syspars.rL1 )
         r = syspars.rL1;
      x = fabs( (Vtarget - V(r, theta, phi)) / Vtarget );
      if( x <= 1.0e-6 )
         break;
   }  
    
   return( r );
}


double TileArea( double r, double theta, struct SphereVector normals,
                 double dtheta, double dphi )
/*********************************
*
*   This function calculates the surface area of the tiles.  A tile has
*   sides with lengths dtheta and dphi.  The area of the tile is set
*   equal to the area of that part of the curved equipotential surface
*   represented by the flat tile.  r is the radius at the center of 
*   the segment.  The vector "normals" must be in spherical polar coords.
*
*********************************/
{
   double nr, dS;

   nr = normals.r;

   dS =   2.0 * r * r * dphi * sin( theta ) * sin( dtheta / 2.0 );
   dS = dS / nr;

   return( dS );
}


double Star2TopY( double x, double z)
/**********************************************************
*
*   This function returns the value of y at the surface of
*   star 2 for (x,z)
*
***********************************************************/
{
   long iteration;
   double converge, Vtarget, r, costheta, theta, cosphi, phi,
          y, yplus, yminus, topy, epsilony, deltay, 
          slope, Vminus, Vzero, Vplus, change;

   converge = 1.0e-6;
   Vtarget = syspars.VL1;

   if( z >= syspars.rL1 ) {
      topy = 0.0;
      return( topy );
   }
   r = sqrt( x*x + z*z );
   if( r > 0.0 ) {
      costheta = z / r;
      theta = acos( costheta );
      phi = 0.0;
      if( x < 0.0 )
         phi = PI;
      Vzero = V( r, theta, phi );
      if( Vzero > Vtarget ) {
         topy = 0.0;
         return( topy );
      }
   }
   
   epsilony = 1.0e-7 * syspars.a;
   y = 0.5 * syspars.rL1;
   for( iteration = 1; ; iteration++) {
      if( iteration > 150 ) {
	 printf("  x = %12.4e  z = %12.4e  y = %12.4e\n", x, z, y);
         Quit("Too many iterations in Star2Topy.");
      }
      
      r = sqrt( x*x + y*y + z*z);
      costheta = z / r;
      theta = acos( costheta );
      cosphi = x / sqrt( x*x + y*y );
      phi = acos( cosphi );
      Vzero  = V( r, theta, phi);
      change = (Vzero - Vtarget) / Vtarget;
      if( fabs( change ) <= converge )
         break;

      yplus = y + epsilony;
      r = sqrt( x*x + yplus*yplus + z*z);
      costheta = z / r;
      theta = acos( costheta );
      cosphi = x / sqrt( x*x + yplus*yplus );
      phi = acos( cosphi );
      Vplus  = V( r, theta, phi);

      yminus = y - epsilony;
      if( yminus < 0.0 )
	 yminus = 0.0;
      r = sqrt( x*x + yminus*yminus + z*z);
      costheta = z / r;
      theta = acos( costheta );
      cosphi = x / sqrt( x*x + yminus*yminus );
      phi = acos( cosphi );
      Vminus = V( r, theta, phi);

      slope = (Vplus - Vminus) / (yplus - yminus);
      if(slope == 0.0)
         slope = 100.0 * Vzero / syspars.a;
      deltay = (Vtarget - Vzero) / slope;
      if( fabs(deltay) > (0.05 * y) ) 
         deltay = (0.05 * y) * ( deltay / fabs( deltay ) );
      y = y + deltay; 
      if( y > (1.01 * Grid.ymax) )
         y = 1.01 * Grid.ymax;
      if( y < 0.0 )
	y = 0.0;
   }
   topy = y; 
    
   return( topy );
}


double GetGDbeta( void )
/***********************************************************
*
*  This function extracts the appropriate value of the gravity
*  darkening beta parameter from the fourbeta[] array, where
*  gravity darkening law is:
*      Teff = <Teff> * ( g / <g> )^beta
*
*********************************************************/
{
   long i, ilow, ihigh;
   double beta, slope;

   if( maxGDindex == 1) {
      beta = 0.25 * fourbeta[1];
      return( beta );      
   }
   if( star2.meanT <= GDT[0] ) {
      beta = 0.25 * fourbeta[1];
      return( beta );      
   }
   if( star2.meanT >= GDT[maxGDindex] ) {
      beta = 0.25 * fourbeta[maxGDindex];
      return( beta );      
   }
   for( i = 0; i < maxGDindex; i++ ) {
      if( (star2.meanT >= GDT[i]) && (star2.meanT < GDT[i+1]) ) {
         ilow = i;
         ihigh = i + 1;
         break;
      }
   }
   if( GDT[ihigh] == GDT[ilow] ) {
      beta = 0.25 * fourbeta[ilow];
      return( beta );      
   }
   slope = ( fourbeta[ihigh] - fourbeta[ilow] ) / ( GDT[ihigh] - GDT[ilow] );
   beta = fourbeta[ilow] + slope * ( star2.meanT - GDT[ilow] );
   beta = 0.25 * beta;

   return( beta );
}


double ClaretHmu( double T, double logg, char *filter, double mu)
/**************************************************************
*
*   This is the Claret (2000, AA, 363, 1081) limb-darkening function.
*   It does not interpolate in logg and T but instead just goes to 
*   the nearest grid point.
*   It does interpolate in mu.
*
*******************************************************************/
{
   long i, Tindex, gindex, findex;
   double deltaT, deltag, a1, a2, a3, a4, sqrtmu, h;

   /*********************************************
   *
   *   First identify the filter index corresponding to the
   *   filter name.
   *
   *********************************************************/
   findex = -1;
   for( i = 0; i <= maxLDfilterindex; i++) {
      if( strcmp( LDfilterName[i], filter) == 0 ) {
         findex = i;
         break;
      }
   }
   if( findex == -1 )
      Quit("Unknown filter name in ClaretHmu.");

   if( T <= LDT[0] ) {
      Tindex = 0;
   }
   else if( T >= LDT[maxLDTindex] ) {
      Tindex = maxLDTindex;
   }
   else {
      deltaT = LDT[1] - LDT[0];
      Tindex = 0.5 + ( T - LDT[0] ) / deltaT;
   }
   if( logg <= LDlogg[0] ) {
      gindex = 0;
   }
   else if( logg >= LDlogg[maxLDgindex] ) {
      gindex = maxLDgindex;
   }
   else {
      deltag =  LDlogg[1] - LDlogg[0];
      gindex = 0.5 + ( logg - LDlogg[0] ) / deltag;
   }

   a1 = LDtable[gindex][Tindex][findex][1];
   a2 = LDtable[gindex][Tindex][findex][2];
   a3 = LDtable[gindex][Tindex][findex][3];
   a4 = LDtable[gindex][Tindex][findex][4];
   if( mu < 0.0 )
      Quit("mu less than zero in ClaretHmu.");
   sqrtmu = sqrt( mu );
   h = 1.0 - a1 * (1.0 - sqrtmu)    - a2 * (1.0 - mu) 
           - a3 * (1.0 - mu*sqrtmu) - a4 * (1.0 - mu*mu);
   return ( h );
}


double GetIperp( double T, double logg, char *filter )
/******************************************************
*
*   This function returns the specific intensity at zero zenith
*   angle by linearly interpolated Iperptable[][][] in temperature
*   and log(g).
*
************************************************************/
{
   long i, findex, Tindex1, Tindex2, gindex1, gindex2;
   double deltaT, deltag, weightT1, weightT2, weightg1, weightg2, intensity;

   /*********************************************
   *
   *   First identify the filter index corresponding to the
   *   filter name.
   *
   *********************************************************/
   findex = -1;
   for( i = 0; i <= maxIperpfilterindex; i++) {
      if( strcmp( IperpfilterName[i], filter) == 0 ) {
         findex = i;
         break;
      }
   }
   if( findex == -1 )
      Quit("Unknown filter name in GetIperp.");

   if( T <= IperpT[0] ) {
      Quit("T out of range (too low) in GetIperp.");
   }
   else if( T >= IperpT[maxIperpTindex] ) {
      Quit("T out of range (too high) in GetIperp.");
   }
   else {
      deltaT = IperpT[1] - IperpT[0];
      Tindex1 = ( T - IperpT[0] + 0.1 ) / deltaT;
      Tindex2 = Tindex1 + 1;
      weightT2 = ( T - IperpT[Tindex1] ) / deltaT;
      weightT1 = 1.0 - weightT2;
   }
   if( logg <= Iperplogg[0] ) {
      gindex1 = 0;
      gindex2 = 0;
      weightg1 = 1.0;
      weightg2 = 0.0;
   }
   else if( logg >= Iperplogg[maxIperpgindex] ) {
      gindex1 = maxIperpgindex;
      gindex2 = maxIperpgindex;
      weightg1 = 0.0;
      weightg2 = 1.0;
   }
   else {
      deltag =  Iperplogg[1] - Iperplogg[0];
      gindex1 = ( logg - Iperplogg[0] + 0.1 ) / deltag;
      gindex2 = gindex1 + 1;
      weightg2 = ( logg - Iperplogg[gindex1] ) / deltag;
      weightg1 = 1.0 - weightg2;
   }
   intensity =  weightg1 * weightT1 * Iperptable[gindex1][Tindex1][findex]
              + weightg1 * weightT2 * Iperptable[gindex1][Tindex2][findex]
              + weightg2 * weightT1 * Iperptable[gindex2][Tindex1][findex]
              + weightg2 * weightT2 * Iperptable[gindex2][Tindex2][findex];

   return( intensity );
}


double Star2L( void )
/**********************************************************
*
*   Calculate the luminosity of star 2 by adding up the fluxes
*   from all the tiles.  This function should not be used until
*   after heating by irradiation has been calculated.
*
************************************************************/
{
   long itile;
   double luminosity;

   luminosity = 0.0;
   for( itile = 1; itile <= star2.Ntiles; itile++) {
      luminosity += SIGMA * pow( T2T[itile], 4.0) * T2dS[itile];
   }

   return( luminosity );
}
