/********************************************
*
*               FILE OUTPUT.C
*
*   All the output functions are in this file.
*
*******************************************/

#include "header.h"

void Quit( char *outputline )
/**********************************************************
*
*   Write a diagnostic comment and quit gracefully from any point
*   in the program.
*
*****************************************************************/
{
   printf("\n   %s\n", outputline);
   printf("   Program terminated.\n");
   exit( 0 );
   return;
}


void WriteResults( void )
/********************************************
*
*   This function writes out the results of the calculations.
*
****************************************************************/
{
   WriteLightCurves();
   WriteSysPars();

   return;
}


void WriteLightCurves( void )
/********************************************
*
*   This function writes out the orbital light curve into the 
*   file litecurves.out
*
****************************************************************/
{
   FILE *out;
   char dummy[80], outputline[300];
   long band, i;
   double lambda1, lambda2;

   if( (out = fopen(filename.lightcurves, "w")) == NULL )
      Quit("Cannot open lightcurve output file .LC");

   if( strcmp( orbit.normalize, "OFF") == 0 ) {
      fprintf( out, " NORMALIZE=        %s\n\n", orbit.normalize);
   }
   else {
      if( strcmp( orbit.normalize, "MAXVALUE") == 0 ) {
         fprintf( out, " NORMALIZE=   %s  %12.4e\n\n", orbit.normalize,
	                                         orbit.normvalue);
      }
      if( strcmp( orbit.normalize, "FITDATA") == 0 ) {
	 if( strcmp( orbit.normfilter, "SQUARE") == 0 ) {
            lambda1 = 1.0e8 * orbit.normMinlambda;
            lambda2 = 1.0e8 * orbit.normMaxlambda;
	    fprintf( out, " NORMALIZE=   %s  %s  %7.1f  %7.1f\n",
		       orbit.normalize, orbit.normfilter, lambda1, lambda2);
         }
         else {
	    fprintf( out, " NORMALIZE=  %s  %s\n",
		                   orbit.normalize, orbit.normfilter);
         }
         fprintf( out, " chi^2 = %12.5e\n\n", data.chisquare);
      }
   }

   sprintf( outputline, "    Orbital ");
   for( band = 1; band <= orbit.nbands; band++ ) {
      if( strcmp( orbit.filter[band], "SQUARE") == 0 ){
         sprintf( dummy, "      %s      ", orbit.filter[band]);
         strcat( outputline, dummy);
      }
      else {
         sprintf( dummy, "                %s              ",
                                  orbit.filter[band]);
         strcat( outputline, dummy);	 
      }
   }
   strcat( outputline, "\n");
   fprintf( out,"%s", outputline);

   sprintf( outputline, "     Phase ");
   for( band = 1; band <= orbit.nbands; band++) {
      if( strcmp( orbit.filter[band], "SQUARE") == 0 ){
         lambda1 = 1.0e8 * orbit.minlambda[band];
         lambda2 = 1.0e8 * orbit.maxlambda[band];
         sprintf( dummy, "      (%4.0f A - %4.0f A) ", lambda1, lambda2);
         strcat( outputline, dummy);
      }
      else {
         strcat( outputline, "                           ");
      }
   }
   strcat( outputline, "\n");
   fprintf( out,"%s", outputline);

   for( i = 0; i <= orbit.maxpindex; i++) {
      sprintf( outputline, "    %7.4f ", LCphase[i]);
      for( band = 1; band <= orbit.nbands; band++ ) {
         sprintf( dummy, " %11.4e  %11.4e", LCflux[band][i], NormLC[band][i]);
         strcat( outputline, dummy);
      }
      strcat( outputline, "\n");
      fprintf( out,"%s", outputline);
   }
   fclose( out );

   return;
}

void WriteSysPars( void )
/***************************************************
*
*   Write various system parameters to a file named syspars.out  
*
***************************************************/
{
   FILE *out;
   long i, band;
   double lambda1, lambda2, x, y, z;

   if( (out = fopen(filename.syspars, "w")) == NULL)
      Quit("Cannot open .SysPars output file.");

   fprintf( out, "STAR1=            %s\n", control.star1);
   fprintf( out, "STAR2=            %s\n", control.star2);
   fprintf( out, "DISK=             %s\n", control.disk);
   fprintf( out, "DISKRIM=          %s\n", control.diskrim);
   fprintf( out, "DISKTORUS=        %s\n", control.disktorus);
   fprintf( out, "INNERDISK=        %s\n", control.innerdisk);
   fprintf( out, "DISKSPOTS=        %s\n", control.diskspots);
   fprintf( out, "ADC=              %s\n", control.adc);
   fprintf( out, "THIRDLIGHT=       %s\n", control.thirdlight);
   fprintf( out, "IRRADIATION=      %s\n", control.irradiation);

   fprintf( out, "\n");
   fprintf( out, "orbit.phasemin      =  %7.4f\n",  orbit.phasemin);
   fprintf( out, "orbit.phasemax      =  %7.4f\n",  orbit.phasemax);
   fprintf( out, "orbit.deltaphase    =  %7.4f\n",  orbit.deltaphase);
   fprintf( out, "orbit.maxpindex     =   %3ld\n",  orbit.maxpindex);
   fprintf( out, "orbit.phaseoffset   =  %7.4f\n",  orbit.phaseoffset);
   fprintf( out, "  Bandpass    Type     Min Wavelength  Max Wavelength\n");
   for( band = 1; band <= orbit.nbands; band++) {
      if( strcmp( orbit.filter[band], "SQUARE") == 0 ) {
	 lambda1 = 1.0e8 * orbit.minlambda[band];
         lambda2 = 1.0e8 * orbit.maxlambda[band];
         fprintf( out, "     %2ld      %s        %5.0f           %5.0f\n",
                   band, orbit.filter[band], lambda1, lambda2);
      }
      else {
	 fprintf( out, "     %2ld         %s\n", band, orbit.filter[band]);
      }
   }
   if( strcmp( orbit.normalize, "OFF") == 0 ) {
      fprintf( out, "NORMALIZE=        %s\n", orbit.normalize);
   }
   else {
      if( strcmp( orbit.normalize, "MAXVALUE") == 0 ) {
         fprintf( out, "NORMALIZE=   %s  %12.4e\n", orbit.normalize,
	                                         orbit.normvalue);
      }
      if( strcmp( orbit.normalize, "FITDATA") == 0 ) {
	 if( strcmp( orbit.normfilter, "SQUARE") == 0 ) {
            lambda1 = 1.0e8 * orbit.normMinlambda;
            lambda2 = 1.0e8 * orbit.normMaxlambda;
	    fprintf( out, "NORMALIZE=   %s  %s  %7.1f  %7.1f\n",
		       orbit.normalize, orbit.normfilter, lambda1, lambda2);
         }
         else {
	    fprintf( out, "NORMALIZE=  %s  %s\n",
		                   orbit.normalize, orbit.normfilter);
        }
      }
   }

   fprintf( out, "\n");
   fprintf( out, "syspars.p           =  %15.8e\n", syspars.p);
   fprintf( out, "syspars.omega       =  %15.8e\n", syspars.omega);
   fprintf( out, "syspars.K2          =  %15.8e\n", syspars.K2);
   fprintf( out, "syspars.q           = %8.4f\n",  syspars.q);
   fprintf( out, "syspars.i           =  %7.4f\n",  syspars.i);
   fprintf( out, "syspars.a           =  %15.8e\n", syspars.a);
   fprintf( out, "syspars.zcm         =  %15.8e\n", syspars.zcm);
   fprintf( out, "syspars.M1 (gm)     =  %10.3e\n", syspars.M1);
   fprintf( out, "syspars.M2 (gm)     =  %10.3e\n", syspars.M2);
   x = syspars.M1 / MSOL;
   fprintf( out, "syspars.M1 (Msun)   =  %6.3f\n",  x);
   x = syspars.M2 / MSOL;
   fprintf( out, "syspars.M2 (Msun)   =  %6.3f\n",  x);
   fprintf( out, "syspars.rL1         =  %15.8e\n", syspars.rL1);
   fprintf( out, "syspars.VL1         =  %15.8e\n", syspars.VL1);
   fprintf( out, "syspars.MeanLobe1Radius =  %7.4f\n", 
                                        syspars.MeanLobe1Radius);
   fprintf( out, "syspars.MeanLobe2Radius =  %7.4f\n", 
                                        syspars.MeanLobe2Radius);

   if( strcmp( control.star1, "ON") == 0) {
      fprintf( out, "\n");
      fprintf( out, "star1.L             =  %10.3e\n", star1.L);
      fprintf( out, "star1.T             =  %10.3e\n", star1.T);
      fprintf( out, "star1.sigmaT4       =  %12.5e\n", star1.sigmaT4);
      fprintf( out, "star1.radius        =  %12.5e\n", star1.radius);
   }

   if( strcmp( control.star2, "ON") == 0) {
      fprintf( out, "\n");
      fprintf( out, "star2.targetNtiles  =   %5ld\n",  star2.targetNtiles);
      fprintf( out, "star2.Ntiles        =   %5ld\n",  star2.Ntiles);
      fprintf( out, "star2.frontradius   =  %12.5e\n", star2.frontradius);
      fprintf( out, "star2.poleradius    =  %12.5e\n", star2.poleradius);
      fprintf( out, "star2.sideradius    =  %12.5e\n", star2.sideradius);
      fprintf( out, "star2.backradius    =  %12.5e\n", star2.backradius);
      fprintf( out, "star2.volume        =  %12.5e\n", star2.volume);
      fprintf( out, "star2.meanr         =  %12.5e\n", star2.meanr);
      fprintf( out, "star2.meang         =  %10.3e\n", star2.meang);
      fprintf( out, "star2.logg          =   %6.3f\n", star2.logg);
      fprintf( out, "star2.meanT         =  %5.0f\n",  star2.meanT);
      fprintf( out, "star2.beta          =   %5.3f\n", star2.beta);
      fprintf( out, "star2.albedo        =   %5.2f\n", star2.albedo);
      fprintf( out, "star2.L             =  %12.5e\n", star2.L);
   }

   if( strcmp( control.star2spots, "ON") == 0) {
      fprintf( out, "\n");
      fprintf( out, "star2spot.nspots = %2ld\n", star2spot.nspots);
      fprintf( out, "   i   theta[i]  phi[i]  radius[i] T-Ratio[i]\n");
      for( i = 1; i <= star2spot.nspots; i++) {
	 x = star2spot.theta[i]  * (360.0 / TWOPI);
         y = star2spot.phi[i]    * (360.0 / TWOPI);
         z = star2spot.radius[i] * (360.0 / TWOPI );
         fprintf( out, "  %2ld   %7.2f  %7.2f  %7.2f     %5.3f\n", 
                     i, x, y, z, star2spot.SpotToverStarT[i] );
      }
   }

   if( strcmp( control.disk, "ON") == 0) {
      fprintf( out, "\n");
      fprintf( out, "disk.targetNtiles   =   %5ld\n",  disk.targetNtiles);
      fprintf( out, "disk.Ntiles         =   %5ld\n",  disk.Ntiles);
      fprintf( out, "disk.e              =   %5.3f\n", disk.e);
      x = disk.zetazero * (360.0/TWOPI);
      fprintf( out, "disk.zetazero       =   %5.1f\n", x);
      fprintf( out, "disk.albedo         =   %5.2f\n", disk.albedo);
      fprintf( out, "disk.L              =  %12.5e\n", disk.L);
      fprintf( out, "disk.TopTmax        =  %10.3e\n", disk.TopTmax);
      fprintf( out, "disk.TopTmin        =  %10.3e\n", disk.TopTmin);

      fprintf( out, "\n");
      fprintf( out, "maindisk.amin       =  %11.4e\n", maindisk.amin);
      fprintf( out, "maindisk.amax       =  %11.4e\n", maindisk.amax);
      fprintf( out, "maindisk.Hmax       =  %11.4e\n", maindisk.Hmax);
      fprintf( out, "maindisk.Hpow       =  %5.2f\n",  maindisk.Hpow);
      fprintf( out, "maindisk.Ttype      =  %s\n",     maindisk.Ttype);
      if( strcmp( maindisk.Ttype, "POWER") == 0) {
         fprintf( out, "maindisk.Tpow       =  %5.2f\n", maindisk.Tpow);
      }
      fprintf( out, "maindisk.maindiskL  =  %10.3e\n", maindisk.maindiskL);
      fprintf( out, "maindisk.Tamax      =  %10.3e\n", maindisk.Tamax);
      fprintf( out, "maindisk.Tamin      =  %10.3e\n", maindisk.Tamin);

      fprintf( out, "\n");
      fprintf( out, "diskedge.T          =   %9.3e\n", diskedge.T);
      fprintf( out, "diskedge.Tspot      =   %9.3e\n", diskedge.Tspot);
      x = diskedge.ZetaMid * (360.0/TWOPI);
      fprintf( out, "diskedge.ZetaMid    =   %5.1f\n",  x);
      x = diskedge.ZetaWidth * (360.0/TWOPI);
      fprintf( out, "diskedge.ZetaWidth  =   %5.1f\n",  x);
   }

   if( strcmp( control.innerdisk, "ON") == 0 ) {
      fprintf( out, "\n");
      fprintf( out, "innerdisk.T         =  %9.2e\n", innerdisk.T);
      fprintf( out, "innerdisk.L         =  %9.2e\n", innerdisk.L);
      fprintf( out, "innerdisk.sigmaT4   =  %9.2e\n", innerdisk.sigmaT4);
      fprintf( out, "innerdisk.radius    =  %9.2e\n", innerdisk.radius);
   }

   if( strcmp( control.diskrim, "ON" ) == 0) {
      fprintf( out, "\n");
      fprintf( out, "diskrim.awidth      =  %11.4e\n", diskrim.awidth);
      fprintf( out, "diskrim.type        =   %s\n",     diskrim.type);
      fprintf( out, "diskrim.Hmax        =  %11.4e\n", diskrim.Hmax);
      fprintf( out, "diskrim.Hmin        =  %11.4e\n", diskrim.Hmin);
      x = diskrim.ZetaHmax * (360.0/TWOPI);
      fprintf( out, "diskrim.ZetaHmax    =   %5.1f\n",  x);
      fprintf( out, "diskrim.Tmax        =   %9.3e\n", diskrim.Tmax);
      fprintf( out, "diskrim.Tmin        =   %9.3e\n", diskrim.Tmin);
      x = diskrim.ZetaTmax * (360.0/TWOPI);
      fprintf( out, "diskrim.ZetaTmax    =   %5.1f\n",  x);
      if( strcmp( diskrim.type, "POINT") == 0 ) {
         fprintf( out, "   i     Zeta[i]       H[i]         T[i]\n");
         for( i = 1; i <= diskrim.points; i++) {
	    x = diskrim.PointZeta[i] * (360.0/TWOPI);
            fprintf( out, "  %2ld     %5.1f     %10.4e    %8.1f\n",
                 i, x, diskrim.PointH[i], diskrim.PointT[i]);
         }
      }
   }

   if( strcmp( control.disktorus, "ON") == 0) {
      fprintf( out, "\n");
      fprintf( out, "disktorus.azero     =  %11.4e\n", disktorus.azero);
      fprintf( out, "disktorus.awidth    =  %11.4e\n", disktorus.awidth);
      fprintf( out, "disktorus.type      =   %s\n",     diskrim.type);
      fprintf( out, "disktorus.Hmax      =  %11.4e\n", disktorus.Hmax);
      fprintf( out, "disktorus.Hmin      =  %11.4e\n", disktorus.Hmin);
      x = disktorus.ZetaHmax * (360.0/TWOPI);
      fprintf( out, "disktorus.ZetaHmax  =   %5.1f\n",  x);
      fprintf( out, "disktorus.Tmax      =   %9.3e\n", disktorus.Tmax);
      fprintf( out, "disktorus.Tmin      =   %9.3e\n", disktorus.Tmin);
      x = disktorus.ZetaTmax * (360.0/TWOPI);
      fprintf( out, "disktorus.ZetaTmax  =   %5.1f\n",  x);
      if( strcmp( disktorus.type, "POINT") == 0 ) {
         fprintf( out, "   i     Zeta[i]       H[i]         T[i]\n");
         for( i = 1; i <= disktorus.points; i++) {
	   x = disktorus.PointZeta[i] * (360.0/TWOPI);
            fprintf( out, "  %2ld     %5.1f     %10.4e    %8.1f\n",
                 i, x, disktorus.PointH[i], disktorus.PointT[i]);
         }
      }
   }

   if( strcmp( control.diskspots, "ON") == 0) {
      fprintf( out, "\n");
      fprintf( out, "diskspot.npoints = %2ld\n", diskspot.nspots);
      fprintf( out, "   i  ZetaMin[i]  ZetaMax[i]   aMin[i]     aMax[i]  spotToverT[i] \n");
      for( i = 1; i <= diskspot.nspots; i++) {
	 x = diskspot.zetamin[i] * (360.0 / TWOPI);
         y = diskspot.zetamax[i] * (360.0 / TWOPI);
         fprintf( out, "  %2ld    %5.1f       %5.1f    %10.4e  %10.4e    %5.2f\n",
                   i, x, y, diskspot.amin[i], diskspot.amax[i], 
                   diskspot.spotToverT[i]);
      }
   }

   if( strcmp( control.adc, "ON") == 0) {
      fprintf( out, "\n");
      fprintf( out, "adc.L               =  %10.3e\n", adc.L);
      fprintf( out, "adc.height          =  %11.4e\n", adc.height);
   }

   if( strcmp( control.thirdlight, "ON") == 0) {
      fprintf( out, "\n");
      fprintf( out, "thirdlight.orbphase =  %6.3f\n",  thirdlight.orbphase);
      fprintf( out, "Third Light Fractions:\n");
      fprintf( out, "  Bandpass  Type   Min Wavelength  Max Wavelength  Fraction\n");
      for( band = 1; band <= thirdlight.nbands; band++) {
         if( strcmp( thirdlight.filter[band], "SQUARE") == 0 ) {
	    lambda1 = 1.0e8 * thirdlight.minlambda[band];
            lambda2 = 1.0e8 * thirdlight.maxlambda[band];
            fprintf( out, "     %2ld    %s       %5.0f           %5.0f        %5.3f\n",
                      band, thirdlight.filter[band], 
                      lambda1, lambda2, thirdlight.fraction[band]);
         }
         else {
	    fprintf( out, "     %2ld       %s                                      %5.3f\n", 
                      band, thirdlight.filter[band],
                      thirdlight.fraction[band]);
         }
      }
    }

   if( data.nbands > 0) {
      fprintf( out, "\n");
      fprintf( out, "Observed Light Curve Data Files:\n");
      fprintf( out, "                      Minimum     Maximum     Data\n");
      fprintf( out, "  Bandpass   Type    Wavelength  Wavelength  Points    Filename\n");
      for( band = 1; band <= data.nbands; band++) {
         if( strcmp( data.filter[band], "SQUARE") == 0 ) {
	    lambda1 = 1.0e8 * data.minlambda[band];
            lambda2 = 1.0e8 * data.maxlambda[band];
            fprintf( out, "     %2ld     %s     %5.0f       %5.0f       %3ld   %s\n",
                      band, data.filter[band], lambda1, lambda2,
                      data.npoints[band], data.filename[band]);
         }
         else {
	    fprintf( out, "     %2ld        %s                               %3ld   %s\n", 
                         band, data.filter[band], 
                         data.npoints[band], data.filename[band]);
         }
      }
   }

   fclose( out );

   return;
}
