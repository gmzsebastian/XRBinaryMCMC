/*****************************************************
*
*                   FILE FITDATA.C
*
*   This file contains functions concerned with fitting the
*   synthetic light curves to observational data.
*
********************************************************/

#include "header.h"

void ReadData( long band )
/*****************************
*
*  This function reads a data lightcurve from the file
*     data.filename[band][]
*   
*  The data are stored in the global structure
*     data.npoints[band]; 
*     data.phase[band][];
*     data.flux[band][];
*     data.standdev[band][];
*
******************************/
{
   void quit();
   FILE *in;
   char inputline[81];
   long int npoints, i, j, nfields, minfields, maxfields;
   double phase, flux, stdev, x;

   if(  (in = fopen( data.filename[band], "r" ))  == NULL ) {
      printf("Cannot open file %s", data.filename[band]); 
      Quit("");
   }

   npoints = 0;
   minfields = 10;
   maxfields = 0;
   while( fgets( inputline, 80, in) != NULL ) {
      npoints = npoints + 1;
      if( npoints > MAXDATAPOINTS )
         Quit("Too many phase points in observed light curve.");
      nfields = sscanf( inputline, "%lf  %lf  %lf", &phase, &flux, &stdev );
      if( nfields < minfields ) minfields = nfields;
      if( nfields > maxfields ) maxfields = nfields;
      if( (phase < -0.5) || (phase > 1.0) ) {
         printf("   Phase of data point %3ld out of range.\n", npoints);
         Quit("");
      }
      data.phase[band][npoints] = phase;
      data.flux[band][npoints] = flux;
      if( nfields >= 3 ) {
         if( stdev <= 0.0 ) {
            printf("   Stand. dev. of data point %3ld out of range.\n",
                                          npoints);
            Quit("");
         }
         data.standdev[band][npoints] = stdev;
      }
      else {
         data.standdev[band][npoints] = 1.0;
      }
   }
   if( npoints <= 0 )
      Quit("No data points in the file containing the observed light curve.");
   data.npoints[band] = npoints;
   if( maxfields != minfields )
      Quit("One or more standard deviations missing in data file.");

   /*******************
   *
   *  Put the points in the observed light curve in order of
   *  increasing orbital phase.
   *
   ********************/
   if( npoints == 1 ) return;
   for( i = 1; i < data.npoints[band]; i++) {
      for( j = i+1; j <= data.npoints[band]; j++) {
         if( data.phase[band][j] < data.phase[band][i] ) {
            x = data.phase[band][i];
            data.phase[band][i] = data.phase[band][j];
            data.phase[band][j] = x;
            x = data.flux[band][i];
            data.flux[band][i] = data.flux[band][j];
            data.flux[band][j] = x;
            x = data.standdev[band][i];
            data.standdev[band][i] = data.standdev[band][j];
            data.standdev[band][j] = x;
         }
      }
   }

   return;
}
