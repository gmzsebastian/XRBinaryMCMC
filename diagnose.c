/*********************************************
*
*               FILE DIAGNOSE.C
*
*   This file contains functions used to diagnose and test the
*   line broadening program.
*
*******************************************/

#include "header.h"

void InspectInput( void )
/*******************************************************
*
*   This function writes out the various input files to files
*   called "inputfilename.inspect".
*
****************************************************************/
{
   long band;

   WritePars();
   WriteGDTable();
   WriteLDTable();
   WriteIperpTable();
   WriteIBBfilterTable();
   WriteZzetaTable();
   if( data.nbands > 0) {
     for( band = 1; band <= data.nbands; band++)
        WriteData( band );
   }

   return;
}


void WritePars( void )
/********************************************
*
*   This function writes out the parameters read from parfile.dat
*   into a file called parfile.inspect.
*
****************************************************************/
{
   FILE *out;
   long i, band;
   char filename[40];

   strcpy(filename, "parfile.inspect");
   if( (out = fopen(filename, "w")) == NULL )
      Quit("Cannot open file parfile.inspect.");
   fprintf( out, "VERBOSE=           %s\n", verbose);
   fprintf( out, "DIAGNOSTICS=       %s  %5.3f  %s\n", control.diagnostics,
                                                      control.diagnosephase,
                                                      control.diagnoseband);

   fprintf( out, "\n");
   fprintf( out, "STAR1=             %s\n", control.star1);
   fprintf( out, "STAR2=             %s\n", control.star2);
   fprintf( out, "DISK=              %s\n", control.disk);
   fprintf( out, "DISKRIM=           %s\n", control.diskrim);
   fprintf( out, "DISKTORUS=         %s\n", control.disktorus);
   fprintf( out, "INNERDISK=         %s\n", control.innerdisk);
   fprintf( out, "DISKSPOTS=         %s\n", control.diskspots);
   fprintf( out, "ADC=               %s\n", control.adc);
   fprintf( out, "THIRDLIGHT=        %s\n", control.thirdlight);
   fprintf( out, "IRRADIATION=       %s\n", control.irradiation);

   fprintf( out, "\n");
   fprintf( out, "PHASES=            %6.3f  %6.3f  %6.3f\n", 
	                   orbit.phasemin, orbit.phasemax, orbit.deltaphase);
   fprintf( out, "PHASEOFFSET=      %7.4f\n",  orbit.phaseoffset);
   for( band = 1; band <= orbit.nbands; band++) {
      if( strcmp( orbit.filter[band], "SQUARE") == 0 ) {
         fprintf( out, "BANDPASS=          SQUARE  %6.0f  %6.0f\n",
                   orbit.minlambda[band], orbit.maxlambda[band]);
      }
      else {
	 fprintf( out, "BANDPASS=          FILTER   %s\n", orbit.filter[band]);
      }
   }
   if( strcmp( orbit.normalize, "OFF") == 0 ) {
      fprintf( out, "NORMALIZE=         %s\n", orbit.normalize);
   }
   else {
      if( strcmp( orbit.normalize, "MAXVALUE") == 0 ) {
         fprintf( out, "NORMALIZE=   %s  %12.4e\n", orbit.normalize,
	                                         orbit.normvalue);
      }
      if( strcmp( orbit.normalize, "FITDATA") == 0 ) {
	if( strcmp( orbit.normfilter, "SQUARE") == 0 ) {
	   fprintf( out, "NORMALIZE=   %s  %s  %7.1f  %7.1f\n",
		             orbit.normalize, orbit.normfilter, 
                             orbit.normMinlambda, orbit.normMaxlambda);
        }
        else {
	  fprintf( out, "NORMALIZE=   %s  %s\n",
		   orbit.normalize, orbit.normfilter);
        }
      }
   }

   fprintf( out, "\n");
   fprintf( out, "PERIOD=            %10.8f\n", syspars.p);
   fprintf( out, "K2=                %6.2f\n",  syspars.K2);
   fprintf( out, "MASSRATIO=         %5.3f\n",  syspars.q);
   fprintf( out, "INCLINATION=       %5.2f\n",  syspars.i);

   if( strcmp( control.star1, "ON") == 0) {
      fprintf( out, "\n");
      fprintf( out, "STAR1LUM=          %9.3e\n", star1.L);
      fprintf( out, "STAR1TEMP=         %9.3e\n", star1.T);
   }

   if( strcmp( control.star2, "ON") == 0) {
      fprintf( out, "\n");
      fprintf( out, "STAR2TILES=       %5ld\n",  star2.targetNtiles);
      fprintf( out, "STAR2TEMP=        %5.0f\n", star2.meanT);
      fprintf( out, "STAR2ALBEDO=      %5.2f\n", star2.albedo);
   }

   if( strcmp( control.disk, "ON") == 0) {
      fprintf( out, "\n");
      fprintf( out, "DISKTILES=         %5ld\n",   disk.targetNtiles);
      fprintf( out, "DISKE=             %5.3f\n",  disk.e);
      fprintf( out, "DISKZETAZERO=     %5.1f\n",  disk.zetazero);
      fprintf( out, "DISKALBEDO=        %5.2f\n", disk.albedo);

      fprintf( out, "\n");
      fprintf( out, "MAINDISKA=         %5.3f  %5.3f\n", maindisk.amin,
	                                              maindisk.amax);
      fprintf( out, "MAINDISKH=         %5.3f  %4.1f\n", maindisk.Hmax,
	                                               maindisk.Hpow);
      fprintf( out, "MAINDISKT=        %5.0f  %4.1f\n", maindisk.Tamax,
	                                              maindisk.Tpow);
      fprintf( out, "\n");
      fprintf( out, "DISKEDGET=        %5.0f  %5.0f  %5.1f  %5.1f\n",
                                      diskedge.T, diskedge.Tspot, 
                                      diskedge.ZetaMid, diskedge.ZetaWidth);
   }

   if( strcmp( control.diskrim, "ON") == 0) {
      fprintf( out, "\n");
      fprintf( out, "DISKRIMAWIDTH=     %5.3f\n",      diskrim.awidth);
      if( strcmp( diskrim.type, "SINUSOID") == 0 )
         fprintf( out, "DISKRIMPARS=       %s   %5.3f %5.3f %5.1f   %6.0f %6.0f %5.1f\n",
	                  diskrim.type, 
                          diskrim.Hmax, diskrim.Hmin, diskrim.ZetaHmax,
                          diskrim.Tmax, diskrim.Tmin, diskrim.ZetaTmax);
      if( strcmp( diskrim.type, "POINT") == 0 ) {
         for( i = 1; i <= diskrim.points; i++) {
            fprintf( out, "DISKRIMPARS=       %s  %5.1f  %5.3f  %7.1f\n",
		     diskrim.type, diskrim.PointZeta[i],
                         diskrim.PointH[i], diskrim.PointT[i]);
         }
      }
    }

   if( strcmp( control.disktorus, "ON") == 0) {
      fprintf( out, "\n");
      fprintf( out, "DISKTORUSAZERO=    %5.3f\n",    disktorus.azero);
      fprintf( out, "DISKTORUSAWIDTH=   %5.3f\n",    disktorus.awidth);
      if( strcmp( disktorus.type, "SINUSOID") == 0 )
         fprintf( out, "DISKTORUSPARS=     %s   %5.3f %5.3f %5.1f   %6.0f %6.0f %5.1f\n",
	                  disktorus.type, 
                          disktorus.Hmax, disktorus.Hmin, disktorus.ZetaHmax,
                          disktorus.Tmax, disktorus.Tmin, disktorus.ZetaTmax);
      if( strcmp( disktorus.type, "POINT") == 0 ) {
         for( i = 1; i <= disktorus.points; i++) {
            fprintf( out, "DISKTORUSPARS=     %s  %5.1f  %5.3f  %7.1f\n",
		     disktorus.type, disktorus.PointZeta[i],
                         disktorus.PointH[i], disktorus.PointT[i]);
         }
      }
    }

   if( strcmp( control.diskspots, "ON") == 0) {
      fprintf( out, "\n");
      for( i = 1; i <= diskspot.nspots; i++) {
         fprintf( out, "DISKSPOT=    %5.1f  %5.1f  %5.3f  %5.3f %6.3f\n",
	             diskspot.zetamin[i], diskspot.zetamax[i],
                     diskspot.amin[i], diskspot.amax[i], diskspot.spotToverT[i]);
      }
   }

   if(strcmp( control.innerdisk, "ON") == 0 ) {
      fprintf( out, "\n");
      fprintf( out, "INNERDISKT=       %9.2e\n", innerdisk.T);
      fprintf( out, "INNERDISKL=       %9.2e\n", innerdisk.L);
   }

   if( strcmp( control.adc, "ON") == 0) {
      fprintf( out, "\n");
      fprintf( out, "ADCL=               =  %10.3e\n", adc.L);
      fprintf( out, "ADCHEIGHT=          =  %11.4e\n", adc.height);
   }

   if( strcmp( control.thirdlight, "ON") == 0) {
      fprintf(out, "\n");
      fprintf( out, "3rdLIGHTPHASE =   %6.3f\n",  thirdlight.orbphase);
      for( i = 1; i <= thirdlight.nbands; i++) {
         if( strcmp( thirdlight.filter[i], "SQUARE") == 0 ) {
	     fprintf( out, "3rdLIGHTFRACTION=  %s %6.0f %6.0f  %5.3f\n",
		        thirdlight.filter[i], thirdlight.minlambda[i],
                        thirdlight.maxlambda[i], thirdlight.fraction[i] );
         }
         else {
	     fprintf( out, "3rdLIGHTFRACTION=  FILTER    %s           %5.3f\n",
		        thirdlight.filter[i], thirdlight.fraction[i] );
         }
      }
   }

   if( data.nbands > 0) {
      fprintf(out, "\n");
      for( i = 1; i <= data.nbands; i++) {
         if( strcmp( data.filter[i], "SQUARE") == 0 ) {
	     fprintf( out, "READDATA=  %s %6.0f %6.0f   %s\n",
		        data.filter[i], data.minlambda[i],
                        data.maxlambda[i], data.filename[i] );
         }
         else {
	     fprintf( out, "READDATA=  FILTER     %s           %s\n",
		        data.filter[i], data.filename[i] );
         }
      }
   }

   fprintf( out, "\nEND\n");

   fclose( out );
   return;
}


void WriteGDTable( void )
/********************************************
*
*   This function writes out the gravity darkening table read 
*   from GDTable.dat into a file called GDTable.inspect.
*
****************************************************************/
{
   FILE *out;
   char filename[40];
   long i;

   strcpy(filename, "GDTable.inspect");
   if( (out = fopen(filename, "w")) == NULL )
      Quit("Cannot open file GDTable.inspect.");

   for( i = 0; i <= maxGDindex; i++) {
     fprintf( out,"  %5.0f  %5.3f\n", GDT[i], fourbeta[i]);
   }

   fclose( out );

   return;
}


void WriteLDTable( void )
/***************************************************
*
*   Write the limb darkening table to the file
*      LDTable.inspect
*
***************************************************/
{
   FILE *out;
   char outfile[30], outputline[200];
   long Tindex, gindex, findex, nfilters;
   double delta;

   strcpy( outfile, "LDTable.inspect" );
   if( (out = fopen(outfile, "w")) == NULL)
      Quit("Cannot open file LDTable.inspect.");

   delta = LDT[1] - LDT[0];
   fprintf( out, "        %5.0f  %5.0f   %4.0f\n", 
                             LDT[0], LDT[maxLDTindex], delta);
   delta = LDlogg[1] - LDlogg[0];
   fprintf( out, "        %4.1f  %4.1f  %4.1f\n",
	                     LDlogg[0], LDlogg[maxLDgindex], delta);
   nfilters = maxLDfilterindex + 1;
   sprintf( outputline, "     %2ld  ", nfilters);
   for( findex = 0; findex <= maxLDfilterindex; findex++ ) {
      strcat( outputline, " ");
      strcat( outputline, LDfilterName[findex]);
   }
   strcat( outputline, "\n");
   fprintf( out,"%s", outputline);

   for( Tindex = 0; Tindex <= maxLDTindex; Tindex++) {
      for( gindex = 0; gindex <= maxLDgindex; gindex++) {
         for( findex = 0; findex <= maxLDfilterindex; findex++) {
	   fprintf( out, " %5.1f  %5.0f  %s   %7.4f   %7.4f   %7.4f   %7.4f\n",
		   LDlogg[gindex], LDT[Tindex], LDfilterName[findex],
			  LDtable[gindex][Tindex][findex][1],
			  LDtable[gindex][Tindex][findex][2],
			  LDtable[gindex][Tindex][findex][3],
			  LDtable[gindex][Tindex][findex][4]);
         }
      }
   }
   fclose( out );

   return;
}


void WriteIperpTable( void )
/***************************************************
*
*   Write the IperpTable to a file named IperpTable.inspect  
*
***************************************************/
{
   FILE *out;
   char outfile[30], outputline[201], dummy[20];
   long Tindex, gindex, findex, nfilters;
   double delta;

   strcpy( outfile, "IperpTable.inspect" );
   if( (out = fopen(outfile, "w")) == NULL)
      Quit("Cannot open file IperpTable.inspect.");

   delta = IperpT[1] - IperpT[0];
   fprintf( out, "     %5.0f  %5.0f   %4.0f\n", 
                             IperpT[0], IperpT[maxIperpTindex], delta);
   delta = Iperplogg[1] - Iperplogg[0];
   fprintf( out, "        %4.1f  %4.1f  %4.1f\n",
	                     Iperplogg[0], Iperplogg[maxIperpgindex], delta);
   nfilters = maxIperpfilterindex + 1;
   sprintf( outputline, "     %2ld  ", nfilters);
   for( findex = 0; findex <= maxIperpfilterindex; findex++ ) {
      strcat( outputline, " ");
      strcat( outputline, IperpfilterName[findex]);
   }
   strcat( outputline, "\n");
   fprintf( out,"%s", outputline);

   for( Tindex = 0; Tindex <= maxIperpTindex; Tindex++) {
      for( gindex = 0; gindex <= maxIperpgindex; gindex++) {
        sprintf( outputline, "%5.0f %4.2f", IperpT[Tindex], Iperplogg[gindex]);
         for( findex = 0; findex <= maxIperpfilterindex; findex++ ) {
            sprintf( dummy, "  %10.3e", Iperptable[gindex][Tindex][findex]);
            strcat( outputline, dummy);
         }
         strcat( outputline, "\n");
         fprintf( out,"%s", outputline);
      }
   }
   fclose( out );

   return;
}


void WriteIBBfilterTable( void )
/**********************************************************
*
*   Write the IBBfilterTable to a file named IBBfilter.inspect
*
**********************************************************/
{
   FILE *out;
   char outfile[40], outputline[201], dummy[20];
   long Tindex, findex, nfilters;
   double delta;

   strcpy( outfile, "IBBfilterTable.inspect" );
   if( (out = fopen(outfile, "w")) == NULL)
      Quit("Cannot open file IBBfilter.inspect.");

   delta = IBBT[1] - IBBT[0];
   fprintf( out, "     %5.0f  %5.0f   %4.0f\n", 
                             IBBT[0], IBBT[maxIBBTindex], delta);
   nfilters = maxIBBfilterindex + 1;
   sprintf( outputline, "     %2ld  ", nfilters);
   for( findex = 0; findex <= maxIBBfilterindex; findex++ ) {
      strcat( outputline, " ");
      strcat( outputline, IBBfilterName[findex]);
   }
   strcat( outputline, "\n");
   fprintf( out,"%s", outputline);

   for( Tindex = 0; Tindex <= maxIBBTindex; Tindex++) {
      sprintf( outputline, "%7.0f", IBBT[Tindex]);
      for( findex = 0; findex <= maxIBBfilterindex; findex++ ) {
         sprintf( dummy, "  %10.3e", IBBtable[Tindex][findex]);
         strcat( outputline, dummy);
      }
      strcat( outputline, "\n");
      fprintf( out,"%s", outputline);
   }
   fclose( out );

   return;
}


void WriteZzetaTable( void )
/***************************************************
*
*   Write the ZBBzeta to a file named ZzetaTable.inspect  
*
***************************************************/
{
   FILE *out;
   char outfile[40];
   long i;
   double BBzeta;

   strcpy( outfile, "ZzetaTable.inspect" );
   if( (out = fopen(outfile, "w")) == NULL)
      Quit("Cannot open file ZzetaTable.inspect.");

   fprintf( out, "  %6ld  %6.4f\n", maxBBzetaindex, deltaBBzeta);
   for( i = 0; i <= maxBBzetaindex; i++) {
     BBzeta = i * deltaBBzeta;
     fprintf( out, " %7.4f   %14.7e\n", BBzeta, ZBBzeta[i]);
   }

   fclose( out );

   return;
}


void InspectStar2Tiles( void )
/*******************************************************
*
*   This function writes out the various properties of the tiles
*   covering star 2.  For convenience of inspection, it writes out
*   the properties into several different files.
*
****************************************************************/
{
   FILE *out;
   char outfile[30], outputline[300], dummy[30];
   long i, band;
   double theta, phi;

   strcpy( outfile, "Star2TilesA.inspect" );
   if( (out = fopen(outfile, "w")) == NULL)
      Quit("Cannot open file Star2TilesA.inspect.");

   fprintf( out, "\n");
   fprintf( out, "  star2.Ntiles   =   %5ld\n",   star2.Ntiles);
   fprintf( out, "  star2.volume   =  %10.3e\n", star2.volume);
   fprintf( out, "  star2.meanr    =  %10.3e\n", star2.meanr);
   fprintf( out, "  star2.meang    =  %10.3e\n", star2.meang);
   fprintf( out, "  star2.logg     =  %6.3f\n",  star2.logg);
   fprintf( out, "  star2.meanT    =  %5.0f\n",  star2.meanT);
   fprintf( out, "  star2.beta     =   %5.3f\n",  star2.beta);
   fprintf( out, "  star2.albedo   =  %5.2f\n",  star2.albedo);

   fprintf( out, "\n");
   fprintf( out, "  Star 2 Tiles:\n\n");
   fprintf( out, "  tile        r         theta     phi          x            y            z\n");
   for( i = 1; i <= star2.Ntiles; i++) {
      theta = T2theta[i] * ( 360.0 / TWOPI );
      phi   = T2phi[i]   * ( 360.0 / TWOPI );
      fprintf( out, "%6ld  %12.5e %8.3f %8.3f   %12.5e %12.5e %12.5e\n",
            i, T2r[i], theta, phi, T2x[i], T2y[i], T2z[i]);
   }

   fclose( out );

   strcpy( outfile, "Star2TilesB.inspect" );
   if( (out = fopen(outfile, "w")) == NULL)
      Quit("Cannot open file Star2TilesB.inspect.");

   fprintf( out, "  Star 2 Tiles:\n\n");
   fprintf( out, "  tile    T2gradV.r   T2gradV.t   T2gradV.p\n");
   for( i = 1; i <= star2.Ntiles; i++) {
      fprintf( out, "%6ld   %10.3e  %10.3e  %10.3e\n",
	       i, T2gradV[i].r, T2gradV[i].theta, T2gradV[i].phi);
   }

   fclose( out );

   strcpy( outfile, "Star2TilesC.inspect" );
   if( (out = fopen(outfile, "w")) == NULL)
      Quit("Cannot open file Star2TilesC.inspect.");

   fprintf( out, "  Star 2 Tiles:\n\n");
   fprintf( out, "  tile normSphere.r normSphere.t normSphere.p normCart.x normCart.y normCart.z\n");
   for( i = 1; i <= star2.Ntiles; i++) {
     fprintf( out, "%6ld  %9.6f     %8.5f     %8.5f   %8.5f   %8.5f   %8.5f\n",
            i, T2normSphere[i].r, T2normSphere[i].theta, T2normSphere[i].phi,
               T2normCart[i].x, T2normCart[i].y, T2normCart[i].z);
   }

   fclose( out );

   strcpy( outfile, "Star2TilesD.inspect" );
   if( (out = fopen(outfile, "w")) == NULL)
      Quit("Cannot open file Star2TilesD.inspect.");

   fprintf( out, "  Star 2 Tiles:\n\n");
   fprintf( out, "  tile        g      log(g)       dS          T\n");
   for( i = 1; i <= star2.Ntiles; i++) {
      fprintf( out, "%6ld   %10.3e %6.3f   %11.4e   %5.0f\n",
            i, T2g[i], T2logg[i], T2dS[i], T2T[i] );
   }

   fclose( out );

   strcpy( outfile, "Star2TilesE.inspect" );
   if( (out = fopen(outfile, "w")) == NULL)
      Quit("Cannot open file Star2TilesE.inspect.");

   fprintf( out, "  Star 2 Tile Specific Intensities:\n\n");

   sprintf( outputline, "     Tile ");
   for( band = 1; band <= orbit.nbands; band++ ) {
      if( strcmp( orbit.filter[band], "SQUARE") == 0 ){
         strcat( outputline, "   ");
         strcat( outputline, orbit.filter[band]);
         strcat( outputline, "   ");
      }
      else {
	 strcat( outputline, "      ");
         strcat( outputline, orbit.filter[band]);
         strcat( outputline, "     ");
      }
   }
   strcat( outputline, "\n");
   fprintf( out,"%s", outputline);

   for( i = 1; i <= star2.Ntiles; i++) {
      sprintf( outputline, "  %6ld ", i);
      for( band = 1; band <= orbit.nbands; band++ ) {
         sprintf( dummy, " %10.3e ", T2I[band][i]);
         strcat( outputline, dummy);
      }
      strcat( outputline, "\n");
      fprintf( out,"%s", outputline);
   }
   fclose( out );

   return;
}


void InspectDiskTiles( double targetarea, 
                       long nringMain, long maintiles, 
                       long nringEdge, long edgetiles )
/*******************************************************
*
*   This function writes out the various properties of the tiles
*   covering star 2.  For convenience of inspection, it writes out
*   the properties into several different files.
*
****************************************************************/
{
   FILE *out;
   char outfile[30], outputline[300], dummy[30];
   long i, band;
   double x, y, zeta;

   strcpy( outfile, "DiskTilesA.inspect" );
   if( (out = fopen(outfile, "w")) == NULL)
      Quit("Cannot open file DiskTilesA.inspect.");

   fprintf( out, "\n");
   fprintf( out, " Disk Tiles:\n");

   fprintf( out, "\n");
   fprintf( out, " disk.targetNtiles  =   %5ld\n",  disk.targetNtiles);
   fprintf( out, " disk.Ntiles        =   %5ld\n",  disk.Ntiles);
   fprintf( out, " disk.e             =   %5.3f\n", disk.e);
   x = disk.zetazero * (360.0/TWOPI);
   fprintf( out, " disk.zetazero      =   %5.1f\n", x);
   fprintf( out, " disk.albedo        =   %5.2f\n", disk.albedo);

   fprintf( out, "\n");
   fprintf( out, " maindisk.amin      =  %11.4e\n", maindisk.amin);
   fprintf( out, " maindisk.amax      =  %11.4e\n", maindisk.amax);
   fprintf( out, " maindisk.Hmax      =  %11.4e\n", maindisk.Hmax);
   fprintf( out, " maindisk.Hpow      =  %5.2f\n",  maindisk.Hpow);
   fprintf( out, " maindisk.Tamax     =  %10.3e\n", maindisk.Tamax);
   fprintf( out, " maindisk.Tamin     =  %10.3e\n", maindisk.Tamin);
   fprintf( out, " maindisk.Tpow      =  %5.2f\n",  maindisk.Tpow);

   fprintf( out, "\n");
   fprintf( out, " diskedge.T         =   %9.3e\n", diskedge.T);
   fprintf( out, " diskedge.Tspot     =   %9.3e\n", diskedge.Tspot);
   x = diskedge.ZetaMid * (360.0/TWOPI);
   fprintf( out, " diskedge.ZetaMid   =   %5.1f\n",  x);
   x = diskedge.ZetaWidth * (360.0/TWOPI);
   fprintf( out, " diskedge.ZetaWidth =   %5.1f\n",  x);

   if( strcmp( control.diskrim, "ON" ) == 0) {
      fprintf( out, "\n");
      fprintf( out, " diskrim.awidth     =  %11.4e\n", diskrim.awidth);
      fprintf( out, " diskrim.type       =   %s\n",     diskrim.type);
      fprintf( out, " diskrim.Hmax       =  %11.4e\n", diskrim.Hmax);
      fprintf( out, " diskrim.Hmin       =  %11.4e\n", diskrim.Hmin);
      x = diskrim.ZetaHmax * (360.0/TWOPI);
      fprintf( out, " diskrim.ZetaHmax   =   %5.1f\n",  x);
      fprintf( out, " diskrim.Tmax       =   %9.3e\n", diskrim.Tmax);
      fprintf( out, " diskrim.Tmin       =   %9.3e\n", diskrim.Tmin);
      x = diskrim.ZetaTmax * (360.0/TWOPI);
      fprintf( out, " diskrim.ZetaTmax   =   %5.1f\n",  x);
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
      fprintf( out, " disktorus.azero    =  %11.4e\n", disktorus.azero);
      fprintf( out, " disktorus.awidth   =  %11.4e\n", disktorus.awidth);
      fprintf( out, " disktorus.type     =   %s\n",     diskrim.type);
      fprintf( out, " disktorus.Hmax     =  %11.4e\n", disktorus.Hmax);
      fprintf( out, " disktorus.Hmin     =  %11.4e\n", disktorus.Hmin);
      x = disktorus.ZetaHmax * (360.0/TWOPI);
      fprintf( out, " disktorus.ZetaHmax =   %5.1f\n",  x);
      fprintf( out, " disktorus.Tmax     =   %9.3e\n", disktorus.Tmax);
      fprintf( out, " disktorus.Tmin     =   %9.3e\n", disktorus.Tmin);
      x = disktorus.ZetaTmax * (360.0/TWOPI);
      fprintf( out, " disktorus.ZetaTmax =   %5.1f\n",  x);
      if( strcmp( disktorus.type, "POINT") == 0 ) {
         fprintf( out, "   i     Zeta[i]       H[i]         T[i]\n");
         for( i = 1; i <= disktorus.points; i++) {
	   x = disktorus.PointZeta[i] * (360.0/TWOPI);
            fprintf( out, "  %2ld     %5.1f     %10.4e    %8.1f\n",
                 i, x, disktorus.PointH[i], disktorus.PointT[i]);
         }
      }
   }

   if( diskspot.nspots >= 1 ) {
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

   fprintf( out, "\n");
   fprintf( out, " targetarea = %12.4e\n", targetarea);
   fprintf( out, " nringMain  = %5ld\n", nringMain);
   fprintf( out, " maintiles  = %5ld\n", maintiles);
   fprintf( out, " nringEdge  = %5ld\n", nringEdge);
   fprintf( out, " edgetiles  = %5ld\n", edgetiles);

   fprintf( out, "\n");
   fprintf( out, "  Tile      a      zeta     rho         h          x          y          z\n");
   for( i = 1; i <= disk.Ntiles; i++) {
      zeta = TDiskZeta[i] * ( 360.0 / TWOPI );
      fprintf( out, "%6ld %10.3e %5.1f %10.3e %10.3e %10.3e %10.3e %10.3e\n",
	       i, TDiska[i] ,zeta, TDiskRho[i], TDiskH[i],
                       TDiskx[i], TDisky[i], TDiskz[i]);
   }
   fclose( out );

   strcpy( outfile, "DiskTilesB.inspect" );
   if( (out = fopen(outfile, "w")) == NULL)
      Quit("Cannot open file DiskTilesB.inspect.");

   fprintf( out, " Disk Tiles:\n\n");
   fprintf( out, " targetarea = %12.4e\n", targetarea);
   fprintf( out, " nringMain  = %5ld\n", nringMain);
   fprintf( out, " maintiles  = %5ld\n", maintiles);
   fprintf( out, " nringEdge  = %5ld\n", nringEdge);
   fprintf( out, " edgetiles  = %5ld\n", edgetiles);

   fprintf( out, "\n");
   fprintf( out, "  Tile  normCyl.rho  normCyl.zeta   normCyl.h  normCart.x normCart.y normCart.z\n");
   for( i = 1; i <= disk.Ntiles; i++) {
    fprintf( out, "%6ld   %9.6f     %8.5f     %8.5f    %8.5f   %8.5f   %8.5f\n",
            i, TDisknormCyl[i].rho, TDisknormCyl[i].zeta, TDisknormCyl[i].h,
               TDisknormCart[i].x,  TDisknormCart[i].y,   TDisknormCart[i].z);
   }
   fclose( out );

   strcpy( outfile, "DiskTilesC.inspect" );
   if( (out = fopen(outfile, "w")) == NULL)
      Quit("Cannot open file DiskTilesC.inspect.");

   fprintf( out, " Disk Tiles:\n\n");
   fprintf( out, " targetarea = %12.4e\n", targetarea);
   fprintf( out, " nringMain  = %5ld\n", nringMain);
   fprintf( out, " maintiles  = %5ld\n", maintiles);
   fprintf( out, " nringEdge  = %5ld\n", nringEdge);
   fprintf( out, " edgetiles  = %5ld\n", edgetiles);

   fprintf( out, "\n");
   fprintf( out, "  Tile      DiskdS         DiskT\n");
   for( i = 1; i <= disk.Ntiles; i++) {
      fprintf( out, "%6ld  %12.4e  %12.4e\n",
	      i, TDiskdS[i], TDiskT[i] );
   }
   fclose( out );

   strcpy( outfile, "DiskTilesD.inspect" );
   if( (out = fopen(outfile, "w")) == NULL)
      Quit("Cannot open file DiskTilesD.inspect.");

   fprintf( out, " Disk Tiles:\n\n");
   fprintf( out, " targetarea = %12.4e\n", targetarea);
   fprintf( out, " nringMain  = %5ld\n", nringMain);
   fprintf( out, " maintiles  = %5ld\n", maintiles);
   fprintf( out, " nringEdge  = %5ld\n", nringEdge);
   fprintf( out, " edgetiles  = %5ld\n\n", edgetiles);

   fprintf( out, "  Disk Tile Specific Intensities:\n\n");

   sprintf( outputline, "     Tile ");
   for( band = 1; band <= orbit.nbands; band++ ) {
      if( strcmp( orbit.filter[band], "SQUARE") == 0 ){
         strcat( outputline, "   ");
         strcat( outputline, orbit.filter[band]);
         strcat( outputline, "   ");
      }
      else {
	 strcat( outputline, "      ");
         strcat( outputline, orbit.filter[band]);
         strcat( outputline, "     ");
      }
   }
   strcat( outputline, "\n");
   fprintf( out,"%s", outputline);

   for( i = 1; i <= disk.Ntiles; i++) {
      sprintf( outputline, "  %6ld ", i);
      for( band = 1; band <= orbit.nbands; band++ ) {
         sprintf( dummy, " %10.3e ", TDiskI[band][i]);
         strcat( outputline, dummy);
      }
      strcat( outputline, "\n");
      fprintf( out,"%s", outputline);
   }
   fclose( out );

   return;
}


void InspectYlimits( void )
/*******************************************************
*
*   This function writes out Grid.Topy[][] and Grid.Bottomy[][].
*
****************************************************************/
{
   FILE *out;
   char outfile[30];
   long ix, iz;
   double x, z;

   strcpy( outfile, "ylimits.inspect" );
   if( (out = fopen(outfile, "w")) == NULL)
      Quit("Cannot open file ylimits.inspect.");

   fprintf( out, "\n");
   fprintf( out, "  Grid.Nxtiles  =  %4ld\n",    Grid.Nxtiles);
   fprintf( out, "  Grid.Nztiles  =  %4ld\n",    Grid.Nztiles);
   fprintf( out, "  Grid.deltax   =  %12.4e\n", Grid.deltax );
   fprintf( out, "  Grid.deltaz   =  %12.4e\n", Grid.deltaz );
   fprintf( out, "  Grid.deltal   =  %12.4e\n", Grid.deltal );
   fprintf( out, "  Grid.xmin     =  %12.4e\n", Grid.xmin   );
   fprintf( out, "  Grid.xmax     =  %12.4e\n", Grid.xmax   );
   fprintf( out, "  Grid.ymin     =  %12.4e\n", Grid.ymin   );
   fprintf( out, "  Grid.ymax     =  %12.4e\n", Grid.ymax   );
   fprintf( out, "  Grid.zmin     =  %12.4e\n", Grid.zmin   );
   fprintf( out, "  Grid.zmax     =  %12.4e\n", Grid.zmax   );

   fprintf( out,"\n");
   fprintf( out,"   ix   iz         x             z           Topy         Bottomy\n");
   for( ix= 1; ix <= Grid.Nxtiles; ix++) {
      x = Grid.xmin + (ix - 1) * Grid.deltax;
      for( iz = 1; iz <= Grid.Nztiles; iz++) {
         z = Grid.zmin + (iz - 1) * Grid.deltaz;
	 fprintf( out,"  %3ld  %3ld   %12.5e  %12.5e  %12.5e  %12.5e\n",
	            ix, iz, x, z, Grid.Topy[ix][iz], Grid.Bottomy[ix][iz]);
      }
   }
   fclose( out );

   return;
}


void InspectEscape( struct CartVector sunvector, char *filter,
                    double Star1Emitted, double Star1Escape,
                    double T2mu[], double T2Emitted[], double T2Escape[],
                    double TDiskmu[], double TDiskEmitted[],
                    double TDiskEscape[] )
/*******************************************************
*
*   This function writes out diagnostic information about the
*   flux emitted from the tiles and whether or not the flux
*   is seen from the Earth.
*
****************************************************************/
{
   FILE *out;
   char outfile[30];
   long itile;

   if( strcmp( control.star1, "ON" ) == 0 ) {
      strcpy( outfile, "escape1.inspect" );
      if( (out = fopen(outfile, "w")) == NULL)
         Quit("Cannot open file escape1.inspect.");

      fprintf( out,"\n");
      fprintf( out, "  sunvector.x  = %7.4f\n", sunvector.x);
      fprintf( out, "  sunvector.y  = %7.4f\n", sunvector.y);
      fprintf( out, "  sunvector.z  = %7.4f\n", sunvector.z);

      fprintf( out,"\n");
      fprintf( out, "  filter       =  %s\n", filter);
      fprintf( out, "  Star1Emitted = %12.5e\n", Star1Emitted);
      fprintf( out, "  Star1Escape  =  %6.4f\n\n", Star1Escape);
      fclose( out );
   }
  
   if( strcmp( control.star2, "ON" ) == 0 ) {
      strcpy( outfile, "escape2.inspect" );
      if( (out = fopen(outfile, "w")) == NULL)
         Quit("Cannot open file escape2.inspect.");

      fprintf( out,"\n");
      fprintf( out, "  sunvector.x  = %7.4f\n", sunvector.x);
      fprintf( out, "  sunvector.y  = %7.4f\n", sunvector.y);
      fprintf( out, "  sunvector.z  = %7.4f\n", sunvector.z);
      fprintf( out, "  filter       =  %s\n", filter);

      fprintf( out,"\n");
      fprintf( out, "                       Emitted    Fraction\n");
      fprintf( out, "  itile       mu         Flux     Escaping\n");
      for( itile = 1; itile <= star2.Ntiles; itile++) {
         fprintf( out,"  %5ld   %8.5f  %12.4e   %6.4f\n",
	        itile, T2mu[itile], T2Emitted[itile], T2Escape[itile] );
      }
      fclose( out );
   }

   if( strcmp( control.disk, "ON" ) == 0 ) {
      strcpy( outfile, "escapeDisk.inspect" );
      if( (out = fopen(outfile, "w")) == NULL)
         Quit("Cannot open file escapeDisk.inspect.");

      fprintf( out,"\n");
      fprintf( out, "  sunvector.x  = %7.4f\n", sunvector.x);
      fprintf( out, "  sunvector.y  = %7.4f\n", sunvector.y);
      fprintf( out, "  sunvector.z  = %7.4f\n", sunvector.z);
      fprintf( out, "  filter       =  %s\n", filter);

      fprintf( out,"\n");
      fprintf( out, "                       Emitted     Fraction\n");
      fprintf( out, "  itile       mu         Flux      Escaping\n");
      for( itile = 1; itile <= disk.Ntiles; itile++) {
         fprintf( out,"  %5ld   %8.5f  %12.4e    %6.4f\n",
	            itile, TDiskmu[itile], TDiskEmitted[itile], 
                    TDiskEscape[itile] );
      }
      fclose( out );
   }

   return;
}


void InspectHeatDiskBy1(double TDiskTold[], double muA1toD[],
                        double DeltaT41toD[], double transmit1toD[] )
/*******************************************************
*
*   This function writes out diagnostic information about the
*   heating of the outer accretion disk by star 1.
*
****************************************************************/
{
   FILE *out;
   char outfile[30];
   long itile;

   strcpy( outfile, "HeatDiskBy1.inspect" );
   if( (out = fopen(outfile, "w")) == NULL)
      Quit("Cannot open file HeatDiskBy1.inspect.");

   fprintf( out, "           Original                           Fraction     Heated\n");
   fprintf( out, "   Tile  Temperature     muA   DeltaT41toDisk  1->Disk   Temperature\n");
   for( itile = 1; itile <= disk.Ntiles; itile++) {
      fprintf( out,"  %5ld  %8.0f     %7.4f   %11.4e    %5.4f    %8.0f\n",
                itile, TDiskTold[itile], muA1toD[itile],
                DeltaT41toD[itile], transmit1toD[itile], TDiskT[itile] );
   }
   fclose( out );

   return;
}


void InspectHeatDiskByID(double TDiskTold[], double muAidtoD[],
                        double DeltaT4idtoD[], double transmitidtoD[] )
/*******************************************************
*
*   This function writes out diagnostic information about the
*   heating of the outer disk by the inner disk.
*
****************************************************************/
{
   FILE *out;
   char outfile[30];
   long itile;

   strcpy( outfile, "HeatDiskByID.inspect" );
   if( (out = fopen(outfile, "w")) == NULL)
      Quit("Cannot open file HeatDiskByID.inspect.");

   fprintf( out, "           Original                           Fraction     Heated\n");
   fprintf( out, "   Tile  Temperature     muA   DeltaT4idtoDisk id->Disk   Temperature\n");
   for( itile = 1; itile <= disk.Ntiles; itile++) {
      fprintf( out,"  %5ld  %8.0f     %7.4f   %11.4e    %5.4f    %8.0f\n",
                itile, TDiskTold[itile], muAidtoD[itile],
                DeltaT4idtoD[itile], transmitidtoD[itile], TDiskT[itile] );
   }
   fclose( out );

   return;
}


void InspectHeatDiskByADC( char *whatside,
                           double TDiskTold[], double muAADCtoD[],
                           double DeltaT4ADCtoD[], double transmitADCtoD[] )
/*******************************************************
*
*   This function writes out diagnostic information about the
*   heating of the outer disk by the ADC.
*
****************************************************************/
{
   FILE *out;
   char outfile[30];
   long itile;

   if( strcmp( whatside, "TOP") == 0) {
      strcpy( outfile, "HeatDiskByTopADC.inspect" );
   }
   else if( strcmp( whatside, "BOTTOM") == 0) {
      strcpy( outfile, "HeatDiskByBotADC.inspect" );
   }
   else {
     Quit("Unrecognized side in function InspectHeatDiskByADC.");
   }
   if( (out = fopen(outfile, "w")) == NULL)
      Quit("Cannot open file HeatDiskByxxxADC.inspect.");

   fprintf( out, "           Original                           Fraction     Heated\n");
   fprintf( out, "   Tile  Temperature     muA   DeltaT4ADC ADC->Disk   Temperature\n");
   for( itile = 1; itile <= disk.Ntiles; itile++) {
      fprintf( out,"  %5ld  %8.0f     %7.4f   %11.4e    %5.4f    %8.0f\n",
                itile, TDiskTold[itile], muAADCtoD[itile],
                DeltaT4ADCtoD[itile], transmitADCtoD[itile], TDiskT[itile] );
   }
   fclose( out );

   return;
}


void InspectHeat2By1( double T2Told[], double muA1to2[], 
                      double DeltaT41to2[], double transmit1to2[] )
/*******************************************************
*
*   This function writes out diagnostic information about the
*   heating of star 2 by star 1.
*
****************************************************************/
{
   FILE *out;
   char outfile[30];
   long itile;

   strcpy( outfile, "Heat2By1.inspect" );
   if( (out = fopen(outfile, "w")) == NULL)
      Quit("Cannot open file Heat2By1.inspect.");

   fprintf( out, "           Original                          Fraction    Heated\n");
   fprintf( out, "   Tile  Temperature     muA    DeltaT41to2    1->2    Temperature\n");
   for( itile = 1; itile <= star2.Ntiles; itile++) {
      fprintf( out,"  %5ld  %8.0f     %7.4f   %11.4e   %5.4f   %8.0f\n",
	         itile, T2Told[itile], muA1to2[itile], 
                 DeltaT41to2[itile], transmit1to2[itile], T2T[itile] );
   }

   fclose( out );

   return;
}


void InspectHeat2ByID( double T2Told[], double muAidto2[], 
                     double DeltaT4idto2[], double transmitidto2[] )
/*******************************************************
*
*   This function writes out diagnostic information about the
*   heating of star 2 by the inner disk.
*
****************************************************************/
{
   FILE *out;
   char outfile[30];
   long itile;

   strcpy( outfile, "Heat2ByInDisk.inspect" );
   if( (out = fopen(outfile, "w")) == NULL)
      Quit("Cannot open file Heat2ByInDisk.inspect.");
   fprintf( out, "           Original                          Fraction    Heated\n");
   fprintf( out, "   Tile  Temperature     muA    DeltaT4idto2   id->2    Temperature\n");
   for( itile = 1; itile <= star2.Ntiles; itile++) {
      fprintf( out,"  %5ld  %8.0f     %7.4f   %11.4e   %5.4f   %8.0f\n",
	         itile, T2Told[itile], muAidto2[itile], 
                 DeltaT4idto2[itile], transmitidto2[itile], T2T[itile] );
   }

   fclose( out );

   return;
}


void InspectHeat2ByDisk( double T2Told[], double DeltaT4Dto2[], double T2T[] )
/*******************************************************
*
*   This function writes out diagnostic information about the
*   heating of star 2 by the disk.
*
****************************************************************/
{
   FILE *out;
   char outfile[30];
   long itile;

   strcpy( outfile, "Heat2ByDisk.inspect" );
   if( (out = fopen(outfile, "w")) == NULL)
      Quit("Cannot open file Heat2ByDisk.inspect.");

   fprintf( out, "           Original                      Heated\n");
   fprintf( out, "   Tile  Temperature    DeltaT4Dto2    Temperature\n");
   for( itile = 1; itile <= star2.Ntiles; itile++) {
      fprintf( out,"  %5ld  %8.0f      %11.4e    %8.0f\n",
	         itile, T2Told[itile], DeltaT4Dto2[itile], T2T[itile] );
   }

   fclose( out );

   return;
}


void InspectHeat2ByADC( char *whatside,
                        double T2Told[], double muAADCto2[],
                        double DeltaT4ADCto2[], double transmitADCto2[] )
/*******************************************************
*
*   This function writes out diagnostic information about the
*   heating of star 2 by the ADC.
*
****************************************************************/
{
   FILE *out;
   char outfile[30];
   long itile;

   if( strcmp( whatside, "TOP") == 0) {
      strcpy( outfile, "Heat2ByTopADC.inspect" );
   }
   else if( strcmp( whatside, "BOTTOM") == 0) {
      strcpy( outfile, "Heat2ByBotADC.inspect" );
   }
   else {
     Quit("Unrecognized side in function InspectHeat2ByADC.");
   }
   if( (out = fopen(outfile, "w")) == NULL)
      Quit("Cannot open file Heat2ByxxxADC.inspect.");

   fprintf( out, "           Original                           Fraction     Heated\n");
   fprintf( out, "   Tile  Temperature     muA     DeltaT4ADC    ADC->2   Temperature\n");
   for( itile = 1; itile <= star2.Ntiles; itile++) {
      fprintf( out,"  %5ld  %8.0f     %7.4f   %11.4e    %5.4f    %8.0f\n",
                itile, T2Told[itile], muAADCto2[itile],
                DeltaT4ADCto2[itile], transmitADCto2[itile], T2T[itile] );
   }
   fclose( out );

   return;
}


void WriteData( long band )
/********************************************
*
*   This function writes out the observed light curves to files
*   called band1data.inspect, band2data.inspect, etc.
*
****************************************************************/
{
   FILE *out;
   char filename[40], bandnum[3];
   long i;

   sprintf( bandnum, "%2ld", band);
   strcpy( filename, "band");
   strcat( filename, bandnum);
   strcat( filename, "data.inspect");
   
   if( (out = fopen(filename, "w")) == NULL ) {
      printf("Cannot open file %s\n", filename);
      Quit("");
   }

   fprintf( out, "Band %2ld from file %s\n", band, data.filename[band]);
   if( strcmp( data.filter[band], "SQUARE") == 0 )
      fprintf( out, "Filter = %s  %5.0f  %5.0f\n", data.filter[band], 
                            data.minlambda[band], data.maxlambda[band]);
   else
      fprintf( out, "Filter = %s\n", data.filter[band]);

   for( i = 1; i <= data.npoints[band]; i++) {
     fprintf( out, " %5.4f  %11.3e  %11.2e\n", data.phase[band][i],
	      data.flux[band][i], data.standdev[band][i]);
   }

   fclose( out );
   
   return;
}
