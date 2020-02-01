/********************************************
*
*               FILE INPUT.C
*
*   All the input functions are in this file.
*
*******************************************/

#include "header.h"

void ReadInput( void )
/***********************************************************
*
*   This is the main function responsible for reading all input
*   files, whether data or parameters.
*
***************************************************************/
{
   if( strcmp( control.diagnostics, "NOCHECKPARS") == 0 ) {
      ReadPars();
   }
   else {
      ReadPars();
      CheckPars();
   }
   ReadGDTable();
   ReadLDTable();
   ReadIperpTable();
   ReadIBBfilterTable();
   ReadZzetaTable();
   if( strcmp( control.diagnostics, "INSPECTINPUT") == 0 ) {
      InspectInput();
      Quit("Quit after INSPECTINPUT.");
   }

   return;
}


void ReadPars( void )
/***************************************************************
*
*   This function reads the input parameters from a file named 
*   "parfile.dat".
*   The file "input.txt" describes the parameter file.                    
*                          
******************************/
{
   char inputline[121], keyword[20], param1[40],
      param2[40], param3[40], param4[40], param5[40], param6[40],
      param7[40], param8[40];
   double x;
   FILE *in;
   long nfields;

   if( (in = fopen(filename.parfile, "r")) == NULL )
      Quit("Cannot open the input parameter file.");

   while(fgets(inputline, 80, in) != NULL){
      nfields = sscanf(inputline, "%s %s %s %s %s %s %s %s %s", keyword,
                param1, param2, param3, param4, param5, 
                param6, param7, param8);

      /***************************************************
      *
      *   Miscellaneous keywords
      *
      *****************************************************/
      if(nfields < 1)
         strcpy(keyword, ""); 
      if(      strcmp(keyword, "END")               == 0)
         break;
      else if( strcmp(keyword, "")                  == 0) 
         x = 1.0; /* blank line */
      else if( strcmp(keyword, "COMMENT=")          == 0)
         x = 1.0; /* comment line */

      /*****************************************************
      *
      *   Do you wonder what the program is doing?
      *
      ******************************************************/
      else if( strcmp(keyword, "VERBOSE=")         == 0)
      	 strcpy( verbose, param1);
      else if( strcmp(keyword, "DIAGNOSTICS=")      == 0) {
         if( nfields < 4 )
	    Quit("Too few parameters for keyword DIAGNOSTICS.");
         sscanf( param1, "%s", control.diagnostics);
         sscanf( param2, "%lf", &control.diagnosephase);
         sscanf( param3, "%s", control.diagnoseband);
      }

      /***************************************************
      *
      *   Keywords controlling the basic properties of the model
      *
      *****************************************************/
      else if( strcmp(keyword, "STAR1=")            == 0)
	 strcpy( control.star1, param1);
      else if( strcmp(keyword, "STAR2=")            == 0)
	 strcpy( control.star2, param1);
      else if( strcmp(keyword, "STAR2SPOTS=")       == 0)
	 strcpy( control.star2spots, param1);
      else if( strcmp(keyword, "DISK=")             == 0)
	 strcpy( control.disk, param1);
      else if( strcmp(keyword, "DISKRIM=")          == 0)
	 strcpy( control.diskrim, param1);
      else if( strcmp(keyword, "DISKTORUS=")        == 0)
	 strcpy( control.disktorus, param1);
      else if( strcmp(keyword, "INNERDISK=")        == 0)
	 strcpy( control.innerdisk, param1);
      else if( strcmp(keyword, "DISKSPOTS=")        == 0)
	 strcpy( control.diskspots, param1);
      else if( strcmp(keyword, "ADC=")              == 0)
	 strcpy( control.adc, param1);
      else if( strcmp(keyword, "THIRDLIGHT=")       == 0)
	 strcpy( control.thirdlight, param1);
      else if( strcmp(keyword, "IRRADIATION=")      == 0)
	 strcpy( control.irradiation, param1);

      /***************************************************
      *
      *   Keywords concerning the orbital light curve.
      *
      *****************************************************/
      else if( strcmp(keyword, "PHASES=") == 0) {
         if( nfields < 4 )
	    Quit("Too few parameters for keyword PHASES.");
         sscanf( param1, "%lf", &orbit.phasemin);
         sscanf( param2, "%lf", &orbit.phasemax);
         sscanf( param3, "%lf", &orbit.deltaphase);
      }
      else if( strcmp(keyword, "PHASEOFFSET=") == 0) {
         sscanf( param1, "%lf", &orbit.phaseoffset);
      }
      else if( strcmp(keyword, "BANDPASS=") == 0) {
	 if( nfields < 3 )
	    Quit("Too few parameters for keyword BANDPASS.");
	 orbit.nbands += 1;
         if( orbit.nbands > (MAXBANDPASSES - 1) )
	    Quit("Too many bandpasses.");
         if( strcmp( param1, "FILTER") == 0) {
	    sscanf( param2, "%s", orbit.filter[orbit.nbands]);
            orbit.minlambda[orbit.nbands] = -1.0;
            orbit.maxlambda[orbit.nbands] = -1.0;
         }
         else if( strcmp( param1, "SQUARE") == 0){
	    if( nfields < 4 )
	       Quit("Too few parameters for BANDPASS= SQUARE.");
            strcpy( orbit.filter[orbit.nbands], "SQUARE");
            sscanf( param2, "%lf", &orbit.minlambda[orbit.nbands]);
            sscanf( param3, "%lf", &orbit.maxlambda[orbit.nbands]);
         }
         else {
	    Quit("BANDPASS: Unrecognized bandpass type.");
         }
      }
      else if( strcmp(keyword, "NORMALIZE=")        == 0) {
         sscanf( param1, "%s", orbit.normalize);
         if( strcmp( orbit.normalize, "MAXVALUE") == 0 ) {
            if( nfields < 3 )
               Quit("Too few parameters for keyword NORMALIZE MAXVALUE.");
            sscanf( param2, "%lf", &orbit.normvalue);
         }
         if( strcmp( orbit.normalize, "FITDATA") == 0 ) {
            if( nfields < 3 )
               Quit("Too few parameters for keyword NORMALIZE FITDATA.");
            sscanf( param2, "%s", orbit.normfilter);
            if( strcmp( orbit.normfilter, "SQUARE") == 0 ) {
               if( nfields < 5 )
                  Quit("Too few parameters for keyword NORMALIZE FITDATA.");
               sscanf( param3, "%lf", &orbit.normMinlambda);
               sscanf( param4, "%lf", &orbit.normMaxlambda); 
            }
         }
      }

      /***************************************************
      *
      *   Keywords concerning the entire system.
      *
      *****************************************************/
      else if( strcmp(keyword, "PERIOD=")           == 0)
         sscanf( param1, "%lf", &syspars.p);
      else if( strcmp(keyword, "K2=")               == 0) 
         sscanf( param1, "%lf", &syspars.K2);
      else if( strcmp(keyword, "M1=")               == 0)
	 sscanf( param1, "%lf", &syspars.M1);
      else if( strcmp(keyword, "MASSRATIO=")        == 0)
         sscanf( param1, "%lf", &syspars.q);
      else if( strcmp(keyword, "INCLINATION=")      == 0)
         sscanf( param1, "%lf", &syspars.i);

      /***************************************************
      *
      *   Keywords concerning the primary star.
      *
      *****************************************************/
      else if( strcmp(keyword, "STAR1LUM=")         == 0)
	 sscanf( param1, "%lf", &star1.L);
      else if( strcmp(keyword, "STAR1TEMP=")        == 0)
	 sscanf( param1, "%lf", &star1.T);

      /***************************************************
      *
      *   Keywords concerning the secondary star.
      *
      *****************************************************/
      else if( strcmp(keyword, "STAR2TILES=")       == 0)
         sscanf( param1, "%ld", &star2.targetNtiles);
      else if( strcmp(keyword, "STAR2TEMP=")        == 0)
         sscanf( param1, "%lf", &star2.meanT);
      else if( strcmp(keyword, "STAR2ALBEDO=")      == 0)
         sscanf( param1, "%lf", &star2.albedo);

      /***************************************************
      *
      *   Keywords concerning star 2 spots.
      *
      *****************************************************/
      else if( strcmp(keyword, "STAR2SPOT=")         == 0) {
         if( nfields < 5 )
	    Quit("Too few parameters in keyword STAR2SPOT.");
	 star2spot.nspots += 1;
         if( star2spot.nspots >= 20 )
	    Quit("Too many star 2 spots.");
         sscanf( param1, "%lf",  &star2spot.theta[star2spot.nspots]);
         sscanf( param2, "%lf",    &star2spot.phi[star2spot.nspots]);
         sscanf( param3, "%lf", &star2spot.radius[star2spot.nspots]);
         sscanf( param4, "%lf", 
                        &star2spot.SpotToverStarT[star2spot.nspots]);
      }

      /***************************************************
      *
      *   Keywords concerning the main accretion disk.
      *
      *****************************************************/
      else if( strcmp(keyword, "DISKTILES=")        == 0)
         sscanf( param1, "%ld", &disk.targetNtiles);
      else if( strcmp(keyword, "DISKE=")            == 0)
	 sscanf( param1, "%lf", &disk.e);
      else if( strcmp(keyword, "DISKZETAZERO=")     == 0)
	 sscanf( param1, "%lf", &disk.zetazero);
      else if( strcmp(keyword, "DISKALBEDO=")       == 0) 
	 sscanf( param1, "%lf", &disk.albedo);

      else if( strcmp(keyword, "MAINDISKA=")        == 0) {
         if( nfields < 3 )
	    Quit("Too few parameters for keyword MAINDISKRHO.");
         sscanf( param1, "%lf", &maindisk.amin);
         sscanf( param2, "%lf", &maindisk.amax);
      }
      else if( strcmp(keyword, "MAINDISKH=")        == 0) {
         if( nfields < 3 )
	    Quit("Too few parameters for keyword MAINDISKH.");
         sscanf( param1, "%lf", &maindisk.Hmax);
         sscanf( param2, "%lf", &maindisk.Hpow);
      }
      else if( strcmp(keyword, "MAINDISKT=")        == 0) {
         if( nfields < 2 )
	    Quit("Too few parameters for keyword MAINDISKT.");
         sscanf( param1, "%s", maindisk.Ttype);
         if( strcmp(maindisk.Ttype, "POWERLAW")     == 0) {
            if( nfields < 4 )
	       Quit("Too few parameters for keyword MAINDISKT.");
            sscanf( param2, "%lf", &maindisk.Tpow);
            sscanf( param3, "%lf", &maindisk.maindiskL);
         }
         else if( strcmp(maindisk.Ttype, "VISCOUS") == 0) {
            if( nfields < 3 )
	       Quit("Too few parameters for keyword MAINDISKT.");
            sscanf( param2, "%lf", &maindisk.maindiskL);
         }
         else
            Quit("Unrecognized temperature distribution for MAINDISKT");
      }

      /***************************************************
      *
      *   Keywords concerning the disk edge.
      *
      *****************************************************/
      else if( strcmp(keyword, "DISKEDGET=")        == 0) {
         if( nfields < 5 )
	    Quit("Too few parameters for keyword DISKRIMT.");
         sscanf( param1, "%lf", &diskedge.T);
         sscanf( param2, "%lf", &diskedge.Tspot);
         sscanf( param3, "%lf", &diskedge.ZetaMid);
         sscanf( param4, "%lf", &diskedge.ZetaWidth);
      }

      /***************************************************
      *
      *   Keywords concerning the innerdisk.
      *
      *****************************************************/
      else if( strcmp(keyword, "INNERDISKT=")        == 0) {
         if( nfields < 2 )
	    Quit("Too few parameters for keyword INNERDISKT.");
         sscanf( param1, "%lf", &innerdisk.T);
      }
      else if( strcmp(keyword, "INNERDISKL=")        == 0) {
         if( nfields < 2 )
	    Quit("Too few parameters for keyword INNERDISKL.");
         sscanf( param1, "%lf", &innerdisk.L);
      }

      /***************************************************
      *
      *   Keywords concerning the disk rim.
      *
      *****************************************************/
      else if( strcmp(keyword, "DISKRIMAWIDTH=")     == 0) {
         sscanf( param1, "%lf", &diskrim.awidth);
      }
      else if( strcmp(keyword, "DISKRIMPARS=")       == 0) {
         if( strcmp( param1, "SINUSOID") == 0 ) {
            if( strcmp( diskrim.type, "MISSING") == 0 ) {
               strcpy( diskrim.type, param1);
            }
	    else if( strcmp( diskrim.type, "SINUSOID") != 0 ) {
	       Quit("DISKRIMPARS: Inconsistent disk rim types.");
            }
            if( nfields < 8 )
	       Quit("Too few parameters in keyword DISKRIMPARS.");
            sscanf( param2, "%lf", &diskrim.Hmax);
            sscanf( param3, "%lf", &diskrim.Hmin);
            sscanf( param4, "%lf", &diskrim.ZetaHmax);
            sscanf( param5, "%lf", &diskrim.Tmax);
            sscanf( param6, "%lf", &diskrim.Tmin);
            sscanf( param7, "%lf", &diskrim.ZetaTmax);
         }
         else if( strcmp( param1, "POINT") == 0 ) {
            if( strcmp( diskrim.type, "MISSING") == 0 ) {
               strcpy( diskrim.type, param1);
            }
            else if( strcmp( diskrim.type, "POINT") != 0 ) {
	       Quit("DISKRIMPARS: Inconsistent disk rim types.");
            }
            if ( nfields < 5 )
	       Quit("Too few parameters in keyword DISKRIMPARS.");
            diskrim.points += 1;
            if( diskrim.points > (MAXZETAPOINTS - 1) )
	       Quit("DISKRIMPARS: Too many points in the POINT rim.");
            sscanf( param2, "%lf", &diskrim.PointZeta[diskrim.points]);
            sscanf( param3, "%lf", &diskrim.PointH[diskrim.points]);
            sscanf( param4, "%lf", &diskrim.PointT[diskrim.points]);
         }
         else
            Quit("DISKRIMPARS: Unrecognized rim type.");
      }

      /***************************************************
      *
      *   Keywords concerning the disk torus.
      *
      *****************************************************/
      else if( strcmp(keyword, "DISKTORUSAZERO=")      == 0) {
         sscanf( param1, "%lf", &disktorus.azero);
      }
      else if( strcmp(keyword, "DISKTORUSAWIDTH=")     == 0) {
         sscanf( param1, "%lf", &disktorus.awidth);
      }
      else if( strcmp(keyword, "DISKTORUSPARS=")       == 0) {
         if( strcmp( param1, "SINUSOID") == 0 ) {
            if( strcmp( disktorus.type, "MISSING") == 0 ) {
               strcpy( disktorus.type, param1);
            }
	    else if( strcmp( disktorus.type, "SINUSOID") != 0 ) {
	       Quit("DISKTORUSPARS: Inconsistent disk torus types.");
            }
            if( nfields < 8 )
	       Quit("Too few parameters in keyword DISKTORUSPARS.");
            sscanf( param2, "%lf", &disktorus.Hmax);
            sscanf( param3, "%lf", &disktorus.Hmin);
            sscanf( param4, "%lf", &disktorus.ZetaHmax);
            sscanf( param5, "%lf", &disktorus.Tmax);
            sscanf( param6, "%lf", &disktorus.Tmin);
            sscanf( param7, "%lf", &disktorus.ZetaTmax);
         }
         else if( strcmp( param1, "POINT") == 0 ) {
            if( strcmp( disktorus.type, "MISSING") == 0 ) {
               strcpy( disktorus.type, param1);
            }
            else if( strcmp( disktorus.type, "POINT") != 0 ) {
	       Quit("DISKTORUSPARS: Inconsistent disk torus types.");
            }
            if( nfields < 5 )
	       Quit("Too few parameters in keyword DISKTORUSPARS.");
            disktorus.points += 1;
            if( disktorus.points > (MAXZETAPOINTS - 1) )
	       Quit("DISKTORUSPARS: Too many points in the POINT torus.");
            sscanf( param2, "%lf", &disktorus.PointZeta[disktorus.points]);
            sscanf( param3, "%lf", &disktorus.PointH[disktorus.points]);
            sscanf( param4, "%lf", &disktorus.PointT[disktorus.points]);
         }
         else
            Quit("DISKTORUSPARS: Unrecognized torus type.");
      }

      /***************************************************
      *
      *   Keywords concerning disk spots.
      *
      *****************************************************/
      else if( strcmp(keyword, "DISKSPOT=")         == 0) {
         if( nfields < 6 )
	    Quit("Too few parameters in keyword DISKSPOT.");
	 diskspot.nspots += 1;
         if( diskspot.nspots >= 20 )
	    Quit("Too many disk spots.");
         sscanf( param1, "%lf", &diskspot.zetamin[diskspot.nspots]);
         sscanf( param2, "%lf", &diskspot.zetamax[diskspot.nspots]);
         sscanf( param3, "%lf", &diskspot.amin[diskspot.nspots]);
         sscanf( param4, "%lf", &diskspot.amax[diskspot.nspots]);
         sscanf( param5, "%lf", &diskspot.spotToverT[diskspot.nspots]);
      }

      /***************************************************
      *
      *   Keywords concerning the ADC.
      *
      *****************************************************/
      else if( strcmp(keyword, "ADCLUM=")         == 0) {
         sscanf( param1, "%lf", &adc.L);
      }
      else if( strcmp(keyword, "ADCHEIGHT=")         == 0) {
         sscanf( param1, "%lf", &adc.height);
      }

      /***************************************************
      *
      *   Keywords concerning the third light.
      *
      *****************************************************/
      else if( strcmp(keyword, "3rdLIGHTPHASE=")    == 0)
         sscanf( param1, "%lf", &thirdlight.orbphase);
      else if( strcmp(keyword, "3rdLIGHTFRACTION=") == 0) {
	 if( nfields < 4 )
	    Quit("Too few parameters for keyword 3rdLIGHTFRACTION.");
	 thirdlight.nbands += 1;
         if( thirdlight.nbands > (MAXBANDPASSES - 1) )
	    Quit("Too many 3rdLIGHT bandpasses.");
         if( strcmp( param1, "FILTER") == 0) {
	    sscanf( param2, "%s", thirdlight.filter[thirdlight.nbands]);
            thirdlight.minlambda[thirdlight.nbands] = -1.0;
            thirdlight.maxlambda[thirdlight.nbands] = -1.0;
            sscanf( param3, "%lf", &thirdlight.fraction[thirdlight.nbands]);
         }
         else if( strcmp( param1, "SQUARE") == 0){
	    if( nfields < 5 )
	       Quit("Too few parameters for BANDPASS= SQUARE.");
            strcpy( thirdlight.filter[thirdlight.nbands], "SQUARE");
            sscanf( param2, "%lf", &thirdlight.minlambda[thirdlight.nbands]);
            sscanf( param3, "%lf", &thirdlight.maxlambda[thirdlight.nbands]);
            sscanf( param4, "%lf", &thirdlight.fraction[thirdlight.nbands]); 
         }
         else {
	    Quit("BANDPASS: Unrecognized bandpass type.");
         }
      }

      /*********************************************************
      *
      *   Keywords concerning the comparison of the synthetic light
      *   curves to observational data.
      *
      **********************************************************/
      else if( strcmp( keyword, "READDATA=") == 0) {
	 if( nfields < 4 )
	    Quit("Too few parameters for keyword READDATA.");
	 data.nbands += 1;
         if( data.nbands > (MAXBANDPASSES - 1) )
	    Quit("Too many data bandpasses.");
         if( strcmp( param1, "FILTER") == 0) {
	    sscanf( param2, "%s", data.filter[data.nbands]);
            data.minlambda[data.nbands] = -1.0;
            data.maxlambda[data.nbands] = -1.0;
            sscanf( param3, "%s", data.filename[data.nbands]);
         }
         else if( strcmp( param1, "SQUARE") == 0){
	    if( nfields < 5 )
	       Quit("Too few parameters for READDATA= SQUARE.");
            strcpy( data.filter[data.nbands], "SQUARE");
            sscanf( param2, "%lf", &data.minlambda[data.nbands]);
            sscanf( param3, "%lf", &data.maxlambda[data.nbands]);
            sscanf( param4, "%s", data.filename[data.nbands]); 
         }
         else {
	    Quit("BANDPASS: Unrecognized bandpass type.");
         }
         data.npoints[data.nbands] = 0;
         ReadData( data.nbands );
      }

      /***************************************************
      *
      *   Oops. Unrecognized keyword in the parameter file.
      *
      *****************************************************/
      else {
         printf("Unrecognized keyword in get_data.\n");
         printf("   keyword =%20s\n", keyword);
         Quit("");
      }
   }
   x = x;
   fclose(in);
   return;
}


void CheckPars( void )
/********************************************************
*
*  This function checks the input parameters to insure that
*  they are reasonable.
*
*********************************************************/
{
   char response[10];
   long i, band, idummy, gridpoints, found;

   if( strcmp( control.diagnostics, "ON") == 0 ) {
      if( (control.diagnosephase < -0.5) || (control.diagnosephase > 1.0) )
         Quit("DIAGNOSTICS: diagnosephase out of range.");
   }

   /******************************************
   * 
   *   Check the parameters dealing with the basic model.  
   *
   ***************************************/
   if( (strcmp( control.star1, "ON") != 0)
           && (strcmp( control.star1, "OFF") != 0) ) {
         Quit("STAR1: neither ON nor OFF");
   }
   if( (strcmp( control.star2, "ON") != 0)
           && (strcmp( control.star2, "OFF") != 0) ) {
         Quit("STAR2: neither ON nor OFF");
   }
   if( (strcmp( control.star2spots, "ON") == 0)
           && (strcmp( control.star2, "OFF") == 0) ) {
         Quit("STAR2SPOTS cannot be ON if STAR2 is OFF.");
   }
   if( (strcmp( control.disk, "ON") != 0)
           && (strcmp( control.disk, "OFF") != 0) ) {
         Quit("DISK: neither ON nor OFF");
   }
   if( (strcmp( control.diskrim, "ON") != 0)
           && (strcmp( control.diskrim, "OFF") != 0) ) {
         Quit("DISKRIM: neither ON nor OFF");
   }
   if( (strcmp( control.diskrim, "ON") == 0)
           && (strcmp( control.disk, "OFF") == 0) ) {
         Quit("DISKRIM cannot be ON if DISK is OFF.");
   }
   if( (strcmp( control.disktorus, "ON") == 0)
           && (strcmp( control.disk, "OFF") == 0) ) {
         Quit("DISKTORUS cannot be ON if DISK is OFF.");
   }
   if( (strcmp( control.disktorus, "ON") != 0)
           && (strcmp( control.disktorus, "OFF") != 0) ) {
         Quit("DISKTORUS: neither ON nor OFF");
   }
   if( (strcmp( control.innerdisk, "ON") != 0)
           && (strcmp( control.innerdisk, "OFF") != 0) ) {
         Quit("INNERDISK: neither ON nor OFF");
   }
   if( (strcmp( control.innerdisk, "ON") == 0)
           && (strcmp( control.disk, "OFF") == 0) ) {
         Quit("INNERDISK cannot be ON if DISK is OFF.");
   }
   if( (strcmp( control.diskspots, "ON") != 0)
           && (strcmp( control.diskspots, "OFF") != 0) ) {
         Quit("DISKSPOTS: neither ON nor OFF");
   }
   if( (strcmp( control.diskspots, "ON") == 0)
           && (strcmp( control.disk, "OFF") == 0) ) {
         Quit("DISKSPOTS cannot be ON if DISK is OFF.");
   }
   if( (strcmp( control.adc, "ON") != 0)
           && (strcmp( control.adc, "OFF") != 0) ) {
         Quit("ADC: neither ON nor OFF");
   }
   if( (strcmp( control.thirdlight, "ON") != 0)
           && (strcmp( control.thirdlight, "OFF") != 0) ) {
         Quit("THIRDLIGHT: neither ON nor OFF");
   }
   if( (strcmp( control.irradiation, "ON") != 0)
           && (strcmp( control.irradiation, "OFF") != 0) ) {
         Quit("IRRADIATION: neither ON nor OFF");
   }

   /******************************************
   * 
   *   Check the parameters dealing with the orbital light curve.  
   *
   ***************************************/
   if( (orbit.phasemin < -0.5) || (orbit.phasemin > 1.0) )
      Quit("PHASES: phasemin out of range.");
   if( (orbit.phasemax < -0.5) || (orbit.phasemax > 1.0) )
      Quit("PHASES: phasemax out of range.");
   if( orbit.phasemin > orbit.phasemax )
     Quit("PHASES: phasemax must be greater than or equal to phasemin.");
   if( (orbit.phaseoffset < -0.5) || (orbit.phaseoffset > 0.5) )
      Quit("PHASEOFFSET: deltaphase must be ge -0.5 and le 0.5.");
   idummy = 1.0 + (orbit.phasemax - orbit.phasemin) / orbit.deltaphase;
   if( idummy > MAXPHASES )
      Quit("Number of orbital phases is greater than MAXPHASES.");
   if( orbit.nbands == 0)
      Quit("No bandpasses specified for the light curves.");
   for( band = 1; band <= orbit.nbands; band++) {
      if( strcmp( orbit.filter[band], "SQUARE") == 0) {
        if( (orbit.minlambda[band] < 0.0) || (orbit.minlambda[band] > 30000.) )
	    Quit("One of the BANDPASS= SQUARE minlambdas is out of range.");
	if( (orbit.maxlambda[band] < 0.0) || (orbit.maxlambda[band] > 30000.) )
	    Quit("One of the BANDPASS= SQUARE maxlambdas is out of range.");
        if( orbit.maxlambda[band] <= orbit.minlambda[band] )
	   Quit("BANDPASS: orbit.minlambda must be le than orbit.maxlambda.");
      }
   }
   if( strcmp( orbit.normalize, "MISSING") == 0 ) {
      Quit("NORMALIZE keyword missing from parfile.dat.");
   }
   if( strcmp( orbit.normalize, "OFF") != 0 ) {
      if( strcmp( orbit.normalize, "MAXVALUE") == 0 ) {
         if( orbit.normvalue <= 0.0 )
	    Quit("NORMALIZE:  normalization value out of range.");
      }
      else if( strcmp( orbit.normalize, "FITDATA") == 0 ) {
         found = 0;
         for( band = 1; band <= orbit.nbands; band++){
	    if( strcmp( orbit.normfilter, orbit.filter[band]) == 0 ) {
	       if( strcmp( orbit.normfilter, "SQUARE") == 0 ) {
		  if( (orbit.normMinlambda == orbit.minlambda[band]) &&
                      (orbit.normMaxlambda == orbit.maxlambda[band]) ) {
		     found = 1;
                  }
               }
               else {
		  found = 1;
               }
            }
         }
         if( found == 0 )
	   Quit("No light curve calculated for FITDATA filter.");
      }
      else {
         Quit("Unrecognized normalization type for keyword NORMALIZE.");
      }
   }

   /******************************************
   * 
   *   Check the parameters referring to the entire system.  
   *
   ***************************************/
   if( (syspars.p < 0.001) || (syspars.p > 365.0) )
      Quit("Orbital period out of range.");
   if( (syspars.K2 < 0.0) && (syspars.M1 < 0.0) )
      Quit("Either M1 or K2 but not both must be specified.");
   if( (syspars.K2 > 0.0) && (syspars.M1 > 0.0) )
      Quit("Either M1 or K2 but not both must be specified.");
   if( syspars.M1 < 0.0 ) {
      if( (syspars.K2 < 1.0) || (syspars.K2 > 1000.0) )
         Quit("K2 out of range.");
   }
   if( syspars.K2 < 0.0 ) {
      if( (syspars.M1 < 0.1) || (syspars.M1 > 100.0) )
         Quit("M1 out of range.");
   }
   if( (syspars.q < 0.01)  ||  (syspars.q > 1.0) )
      Quit("Mass ratio must lie between 0.01 and 1.00.");
   if( (syspars.i <= 0.0)  ||  (syspars.i > 90.0) )
      Quit("Inclination must be gt 0.0 and le 90.0.");

   /******************************************
   * 
   *   Check the parameters referring to star 1.
   *
   ***************************************/
   if( strcmp(control.star1, "ON") == 0 ) {
      if( (star1.L < 0.0) || (star1.L > 1.0e40) )
         Quit("Luminosity of star 1 out of range.");
      if( (star1.T < 0.0) || (star1.T > 3.0e7) )
         Quit("Temperature of star 1 out of range.");
   }

   /******************************************
   * 
   *   Check the parameters referring to star 2.
   *
   ***************************************/
   if( (star2.targetNtiles < 100) || (star2.targetNtiles > 0.99 * MAX2TILES) )
      Quit("STAR2TILES must be ge 100 and le 0.99*MAX2TILES.");
   if( (star2.meanT < 3.0) || (star2.meanT > 1.0e4) )
      Quit("Temperature of star 2 out of range.");
   if( (star2.albedo < 0.0) || (star2.albedo > 1.0) )
      Quit("STAR2ALBEDO out of range.");

   /******************************************
   * 
   *   Check the parameters referring to the star 2 spots.
   *
   ***************************************/
   if( strcmp( control.star2spots, "ON") == 0 ) {
      if( star2spot.nspots <= 0) 
         Quit("STAR2SPOTS= ON, but no spots specified.");
      for( i = 1; i <= star2spot.nspots; i++ ) {
         if( (star2spot.theta[i] < 0.0) || (star2spot.theta[i] > 180.0) )
            Quit("STAR2SPOT: theta out of range.");
         if( (star2spot.phi[i] < 0.0) || (star2spot.phi[i] > 360.0) )
            Quit("STAR2SPOT: phi out of range.");
         if( (star2spot.radius[i] <= 0.0) || (star2spot.radius[i] > 90.0) )
            Quit("STAR2SPOT: radius out of range.");
         if( (star2spot.SpotToverStarT[i] <= 0.0)
                || (star2spot.SpotToverStarT[i] > 2.0) )
            Quit("STAR2SPOT: SpotToverStarT out of range.");
      }
   }

   /******************************************
   * 
   *   Check the parameters referring to the main part of the disk.
   *
   ***************************************/
   if( strcmp( control.disk, "ON") == 0 ) {
      if( (disk.targetNtiles < 100) || (disk.targetNtiles > 0.99 * MAXDISKTILES) )
         Quit("DISKTILES must be ge 100 and le 0.99*MAXDISKTILES.");
      if( (disk.e < 0.0) || (disk.e >= 1.0) )
         Quit("DISKE: disk.e out of range.");
      if( (disk.zetazero < 0.0) || (disk.zetazero >= 360.0) )
         Quit("DISKZETAZERO: disk.zetazero out of range.");
      if( (disk.albedo < 0.0) || (disk.albedo > 1.0) )
	 Quit("DISKALBEDO out of range.");

      if( (maindisk.amin <0.0) || (maindisk.amin > 0.6) )
         Quit("MAINDISKA: maindisk.amin out of range.");
      if( (maindisk.amax <0.0) || (maindisk.amax > 0.6) )
         Quit("MAINDISKA: maindisk.amax out of range.");
      if( maindisk.amax <= maindisk.amin )
         Quit("MAINDISKA: maindisk.amax must be greater than maindisk.amin.");
      if( (maindisk.Hmax <= 0.0) || (maindisk.Hmax > maindisk.amax) )
         Quit("MAINDISKH: maindisk.Hmax must be gt 0.0 and le maindisk.amax.");
      if( (maindisk.Hpow < 0.0) || (maindisk.Hpow > 2.0) )
         Quit("MAINDISKH: maindisk.Hpow must be between 0.0 and 2.0.");
      if( (maindisk.maindiskL < 1.0e28) || (maindisk.maindiskL > 1.0e39))
         Quit("MAINDISKT: maindisk luminosity must be between 1.0e28 and 1.0e39");
      if( strcmp(maindisk.Ttype, "POWERLAW")     == 0) {
         if( (maindisk.Tpow < -3.0) || (maindisk.Tpow > 3.0) )
            Quit("MAINDISKT: maindisk.Tpow must be between -3.0 and 3.0.");
      }
      if( (diskedge.T < 0.0) || (diskedge.T > 1.0e6) )
         Quit("DISKEDGET: Edge T out of range.");
      if( (diskedge.Tspot < 0.0) || (diskedge.Tspot > 1.0e6) )
         Quit("DISKEDGET: Tspot out of range.");
      if( (diskedge.ZetaMid < 0.0) || (diskedge.ZetaMid >= 360.0) )
         Quit("DISKEDGET: ZetaMid must be ge 0.0 and lt 360.0.");
      if( (diskedge.ZetaWidth < 0.0) || (diskedge.ZetaWidth >= 360.0) )
         Quit("DISKEDGET: ZetaWidth must be ge 0.0 and lt 360.0.");
   }

   /******************************************
   * 
   *   Check the parameters referring to the inner disk.
   *
   ***************************************/
   if( strcmp( control.innerdisk, "ON") == 0 ) {
      if( (innerdisk.T < 0.0) || (innerdisk.T > 1.0e7) )
	Quit("Inner disk temperature out of range.");
      if( (innerdisk.L < 500.0) || (innerdisk.L > 1.0e39) )
	  Quit("Inner disk luminosity out of range.");
   }

   /******************************************
   * 
   *   Check the parameters referring to the disk rim.
   *
   ***************************************/
   if( strcmp( control.diskrim, "ON") == 0 ) {
      if( diskrim.awidth <= 0.0 )
         Quit("DISKRIMAWIDTH is out of range.");
      if( diskrim.awidth > (maindisk.amax - maindisk.amin) )
         Quit("DISKRIMAWIDTH is greater than the disk width.");
      if( strcmp( diskrim.type, "SINUSOID") == 0 ) {
         if( (diskrim.Hmax < 0.0) || (diskrim.Hmax > maindisk.amax) )
            Quit("DISKRIMPARS: Hmax must be ge 0.0 and le maindisk.amax.");
         if( (diskrim.Hmin < 0.0) || (diskrim.Hmin > maindisk.amax) )
            Quit("DISKRIMPARS: Hmin must be ge 0.0 and le maindisk.amax.");
         if( diskrim.Hmax < diskrim.Hmin )
            Quit("DISKRIMPARS: Hmax must be ge Hmin.");
         if( (diskrim.ZetaHmax < 0.0) || (diskrim.ZetaHmax >= 360.0) )
            Quit("DISKRIMPARS: ZetaHmax must be ge 0.0 and lt 360.0.");
         if( (diskrim.Tmax < 0.0) || (diskrim.Tmax > 1.0e6) )
            Quit("DISKRIMPARS: Tmax out of range.");
         if( (diskrim.Tmin < 0.0) || (diskrim.Tmin > 1.0e6) )
            Quit("DISKRIMPARS: Tmin out of range.");
         if( diskrim.Tmax < diskrim.Tmin )
            Quit("DISKRIMPARS: Tmax must be ge than Tmin.");
         if( (diskrim.ZetaTmax < 0.0) || (diskrim.ZetaTmax >= 360.0) ) 
	    Quit("DISKRIMPARS: ZetaTmax must be ge 0.0 and lt 360.0.");
      }
      else if( strcmp( diskrim.type, "POINT") == 0 ) {
         for( i = 1; i <= diskrim.points; i++) {
	    if( (diskrim.PointZeta[i] < 0.0) ||
                           (diskrim.PointZeta[i] >= 360.0) )
	       Quit("DISKRIMPARS: The PointZetas must be ge 0.0 and lt 360.0.");
            if( (diskrim.PointH[i] < 0.0) || 
                           (diskrim.PointH[i] > maindisk.amax) )
 	       Quit("DISKRIMPARS: Rim H must be ge 0 and le maindisk.amax.");
            if( (diskrim.PointT[i] < 0.0) || 
                           (diskrim.PointT[i] > 1.0e6) )
               Quit("DISKRIMPARS: At least one rim T is out of range.");
         }
      }
      else
         Quit("DISKRIMPARS: Unrecognized type.");
   }

   /******************************************
   * 
   *   Check the parameters referring to the disk torus.
   *
   ***************************************/
   if( strcmp( control.disktorus, "ON") == 0 ) {
      if( (disktorus.azero >= maindisk.amax) 
                  || (disktorus.azero <= maindisk.amin) )
	 Quit("DISKTORUSAZERO is outside the disk.");
      if( (disktorus.azero - 0.5 * disktorus.awidth) < maindisk.amin)
	 Quit("DISKTORUSAWIDTH: torus extends past the disk inner edge.");
      if( (disktorus.azero + 0.5 * disktorus.awidth) > maindisk.amax)
         Quit("DISKTORUSAWIDTH: torus extends past the disk outer edge.");
      if( strcmp( control.diskrim, "ON") == 0) {
         if( (maindisk.amax - diskrim.awidth) 
	            < (disktorus.azero + 0.5 * disktorus.awidth) ) {
	    Quit("Disk rim and disk torus overlap.");
         }
      }
      if( strcmp( disktorus.type, "SINUSOID") == 0 ) {
         if( (disktorus.Hmax < 0.0) || (disktorus.Hmax > maindisk.amax) )
            Quit("DISKTORUSPARS: Hmax must be ge 0.0 and le maindisk.amax.");
         if( (disktorus.Hmin < 0.0) || (disktorus.Hmin > maindisk.amax) )
            Quit("DISKTORUSPARS: Hmin must be ge 0.0 and le maindisk.amax.");
         if( disktorus.Hmax < disktorus.Hmin )
            Quit("DISKTORUSPARS: Hmax must be ge Hmin.");
         if( (disktorus.ZetaHmax < 0.0) || (disktorus.ZetaHmax >= 360.0) )
            Quit("DISKTORUSPARS: ZetaHmax must be ge 0.0 and lt 360.0.");
         if( (disktorus.Tmax < 0.0) || (disktorus.Tmax > 1.0e6) )
            Quit("DISKTORUSPARS: Tmax out of range.");
         if( (disktorus.Tmin < 0.0) || (disktorus.Tmin > 1.0e6) )
            Quit("DISKTORUSPARS: Tmin out of range.");
         if( disktorus.Tmax < disktorus.Tmin )
            Quit("DISKTORUSPARS: Tmax must be ge than Tmin.");
         if( (disktorus.ZetaTmax < 0.0) || (disktorus.ZetaTmax >= 360.0) ) 
	    Quit("DISKTORUSPARS: ZetaTmax must be ge 0.0 and lt 360.0.");
      }
      else if( strcmp( disktorus.type, "POINT") == 0 ) {
         for( i = 1; i <= disktorus.points; i++) {
	    if( (disktorus.PointZeta[i] < 0.0) ||
                           (disktorus.PointZeta[i] >= 360.0) )
	       Quit("DISKTORUSPARS: The PointZetas must be ge 0.0 and lt 360.0.");
            if( (disktorus.PointH[i] < 0.0) || 
                           (disktorus.PointH[i] > maindisk.amax) )
 	       Quit("DISKTORUSPARS: Torus H must be ge 0 and le maindisk.amax.");
            if( (disktorus.PointT[i] < 0.0) || 
                           (disktorus.PointT[i] > 1.0e6) )
               Quit("DISKTORUSPARS: At least one torus T is out of range.");
         }
      }
      else
         Quit("DISKTORUSPARS: Unrecognized type.");
   }

   /******************************************
   * 
   *   Check the parameters referring to the disk spots.
   *
   ***************************************/
   if( strcmp( control.diskspots, "ON") == 0 ) {
      if( diskspot.nspots <= 0) 
         Quit("DISKSPOTS= ON, but no spots specified.");
      for( i = 1; i <= diskspot.nspots; i++ ) {
         if( (diskspot.zetamin[i] < 0.0) || (diskspot.zetamin[i] > 360.0) )
            Quit("diskspot.zetamin out of range.");
         if( (diskspot.zetamax[i] < 0.0) || (diskspot.zetamin[i] > 360.0) )
            Quit("diskspot.zetamax out of range.");
         if( diskspot.amin[i] >= diskspot.amax[i] )
            Quit("diskspot.amax must be greater than diskspot.amin.");
         if( (diskspot.amin[i] < 0.0) || (diskspot.amin[i] > 0.6) )
            Quit("diskspot.amin out of range.");
         if( (diskspot.amax[i] < 0.0) || (diskspot.amax[i] > 0.6) )
            Quit("diskspot.amin out of range.");
         if( (diskspot.spotToverT[i] < 0.0) 
                || (diskspot.spotToverT[i] > 100.0) )
            Quit("diskspot.spotToverT out of range.");
      }
   }

   /******************************************
   * 
   *   Check the parameters referring to the ADC.
   *
   ***************************************/
   if( strcmp( control.adc, "ON") == 0 ) {
      if( (adc.L < 0.0) || (adc.L > 1.0e40) )
         Quit("Luminosity of the ADC is out of range.");
      if( (adc.height <= 0.0) || (adc.height > 0.5) )
         Quit("Height of the point-approx ADC is out of range.");
   }

   /*******************************************************
   *
   *  Check third light commands
   *
   ********************************************************/
   if( strcmp( control.thirdlight, "ON" ) == 0 ) {
      if( (thirdlight.orbphase < -0.5) || (thirdlight.orbphase >= 1.0) )
         Quit("3rdLIGHTPHASE out of range.");
      if( thirdlight.nbands <= 0 )
         Quit("Third light is on but no 3rdLIGHTFRACTIONs specified.");
      for( i = 1; i <= thirdlight.nbands; i++) {
         if( strcmp( orbit.filter[band], "SQUARE") == 0) {
            if( (thirdlight.minlambda[i] < 0.0) 
                              || (thirdlight.minlambda[i] > 30000.) )
	       Quit("One of the 3rd light SQUARE minlambdas is out of range.");
            if( (thirdlight.maxlambda[i] < 0.0) 
                              || (thirdlight.maxlambda[i] > 30000.) )
	       Quit("One of the 3rd light SQUARE maxlambdas is out of range.");
            if( thirdlight.maxlambda[i] <= orbit.minlambda[i] )
	       Quit("The 3rd light orbit.minlambda must be le than orbit.maxlambda.");
         }
         if( (thirdlight.fraction[i] < 0.0) 
                              || (thirdlight.fraction[i] >= 1.0) )
            Quit("3rd light fraction must be ge 0.0 and lt 1.0.");
      }
   }

   if( (strcmp( verbose, "ON") != 0)
           && (strcmp( verbose, "OFF") != 0) ) {
         Quit("VERBOSE must be ON or OFF.");
   }
   return;
}


void ReadGDTable( void )
/*****************************************************
*
*   This function read the table containing the gravity
*   darkening coefficients.
*   Note that this function assumes that the
*   coefficients in the table are 4*beta, where 
*      Teff = <Teff> * ( g / <g> )^beta
*
*   The gravity darkening data are stored in the global variables
*     long maxGDindex
*     double GDT[], fourbeta[]
*   
**********************************************************/
{
   FILE *in;
   char filename[40], inputline[81];
   long i;

   strcpy(filename, "GDTable.dat");
   if((in = fopen(filename, "r")) == NULL)
      Quit("Cannot open file GDTable.dat.");

   i = -1;
   while(fgets(inputline, 80, in) != NULL){
      if( inputline[0] != '*' ) {
         i = i +1;
         sscanf(inputline, "%lf %lf", &GDT[i], &fourbeta[i] );
      }
   }
   maxGDindex = i;
   fclose ( in );

   return;
}


void ReadLDTable( void )
/*****************************************************************
*
*   This function reads the limb darkening table LDTable.dat.   
*   The limb darkening law is the four-parameter law advocated by
*   A. Claret (2000, A&A, 363, 1081):
*
*        I(mu) / I(1) =  1 - a1*(1 - mu**(1/2))
*                          - a2*(1 - mu**(1)  )
*                          - a3*(1 - mu**(3/2))
*                          - a4*(1 - mu**(2)  )
*
*   The file must have the format:
*       
*       Tmin    Tmax    deltaT    =  The minimum and maximum temperature
*                                    and the temperature spacing.
*       gmin    gmax    deltag    =  The minimum and maximum LOG g
*                                    and the spacing in log g.
*         N filtername1 filtername2 filtername3 ... filternameN
*        T   log(g)   lambda    a1    a2    a3    a4
*        .     .        .       .     .     .     .
*        .     .        .       .     .     .     .
*        .     .        .       .     .     .     .
*
*
*   The limb darkening table is stored in the global variables
*      long maxgindex, maxTindex, maxlindex
*      double LDlogg[gindex], LDT[Tindex], LDlambda[lindex]
*      double LDtable[gindex][Tindex][lindex][aindex]
*  
*   The maximum values of the indices in these arrays is set
*   in header.h.  Be careful.
*
*******************************************************************/
{
   FILE *in;
   char filename[40], inputline[201], filtername[20];
   long i, nfilters, Tindex, gindex, findex;
   double Tmin, Tmax, deltaT, gmin, gmax, deltag, logg, T, a[5];

   strcpy(filename, "LDTable.dat");
   if((in = fopen(filename, "r")) == NULL)
      Quit("Cannot open file LDTable.dat.");

   /******************************************************
   *
   *   Skip over any initial comment lines (lines beginning
   *   with a '*') then read three lines containing the
   *   information about the contents of the table.
   *
   ********************************************************/
   for(;;) {
      fgets(inputline, 80, in);
      if( inputline[0] != '*' ) {
            sscanf( inputline, "%lf %lf %lf", &Tmin, &Tmax, &deltaT);
            maxLDTindex = ( Tmax - Tmin + 0.1 ) / deltaT;
            for( i = 0; i <= maxLDTindex; i++) {
	       LDT[i] = Tmin + i * deltaT;
            }
         fgets(inputline, 80, in);
            sscanf( inputline, "%lf %lf %lf", &gmin, &gmax, &deltag);
            maxLDgindex = ( gmax - gmin + 0.001) / deltag;
            for( i = 0; i <= maxLDgindex; i++) {
	       LDlogg[i] = gmin + i * deltag;
            }
         fgets(inputline, 80, in);
            sscanf( inputline, "%ld %s %s %s %s %s %s %s %s %s %s %s %s", 
		      &nfilters,
                      LDfilterName[0], LDfilterName[1],  LDfilterName[2],
                      LDfilterName[3], LDfilterName[4],  LDfilterName[5],
                      LDfilterName[6], LDfilterName[7],  LDfilterName[8],
		      LDfilterName[9], LDfilterName[10], LDfilterName[11]);
            maxLDfilterindex = nfilters - 1;
            if( maxLDfilterindex > (MAXFILTERS - 1) )
	      Quit("Too many filters in the LD table.");
         break;
      }
   }

   while( fscanf( in , "%lf %lf %s %lf %lf %lf %lf",
	  &logg, &T, filtername, &a[1], &a[2], &a[3], &a[4] ) != EOF ) {
      gindex = (logg - gmin + 0.1) / deltag;
      if( (gindex < 0) || (gindex > maxLDgindex) )
         Quit("gindex out of range in ReadLDTable.");
      Tindex = ( T - Tmin + 1.0) / deltaT;
      if( (Tindex < 0) || (Tindex > maxLDTindex) )
         Quit("Tindex out of range in ReadLDTable.");
      findex = -1;
      for( i = 0; i <= maxLDfilterindex; i++ ) {
         if( strcmp( filtername, LDfilterName[i]) == 0 ) 
            findex = i;
      }
      if( (findex < 0) || (findex > maxLDfilterindex) )
         Quit("Unrecognized filter name in ReadLDTable.");
      for( i = 1; i <= 4; i++ )
            LDtable[gindex][Tindex][findex][i] = a[i];
   }
   fclose ( in );
   return;
}


void ReadIperpTable()
/*****************************************************************
*
*   This function reads IperpTable.dat.   The file containing the Iperp
*   data must have the format:
*       Tmin    Tmax    deltaT    =  The minimum and maximum temperature
*                                    and the temperature spacing.
*       gmin    gmax    deltag    =  The minimum and maximum LOG g
*                                    and the spacing in log g.
*         n    filter1  filter2 filter3 filter4 ........
*        t   logg    Iz(lambda1)  Iz(lambda2)  Iz(lambda3)  Iz(lambda4)......
*        .     .           .            .            .            .
*        .     .           .            .            .            .
*        .     .           .            .            .            .
*
*    n is the number of filters.  "filter1" "filter2", etc. are the
*      names of the filters, eg U B V R I
*
*   The Iperp data are stored in the global variables
*      long maxIperpgindex, maxIperpTindex, maxIperplindex
*      double Iperplogg[gindex], IperpT[Tindex], Iperplambda[lindex]
*      double Iperptable[gindex][Tindex][lindex]
*  
*******************************************************************/
{
   FILE *in;
   char filename[40], inputline[201], Iperpfilter[15][20];
   long Tindex, gindex, findex, nfilters;
   double Tmin, Tmax, deltaT, gmin, gmax, deltag, xlogg, xT, xIperp[20];

   strcpy(filename, "IperpTable.dat");
   if((in = fopen(filename, "r")) == NULL)
      Quit("Cannot open file IperpTable.dat.");

   fscanf( in, "%lf  %lf  %lf", &Tmin, &Tmax, &deltaT);
      maxIperpTindex = ( Tmax - Tmin + 1.0 ) / deltaT;
      for( Tindex = 0; Tindex <= maxIperpTindex; Tindex++) {
	 IperpT[Tindex] = Tmin + Tindex * deltaT;
      }
   fscanf( in, "%lf  %lf  %lf", &gmin, &gmax, &deltag);
      maxIperpgindex = ( gmax - gmin + 0.01) / deltag;
      for( gindex = 0; gindex <= maxIperpgindex; gindex++) {
         Iperplogg[gindex] = gmin + gindex * deltag;
      }
   fgets(inputline, 120, in);
   fgets(inputline, 120, in);
   sscanf( inputline, "%ld %s %s %s %s %s %s %s %s %s %s %s %s",
            &nfilters,
            IperpfilterName[0],  IperpfilterName[1],  IperpfilterName[2], 
            IperpfilterName[3],  IperpfilterName[4],  IperpfilterName[5],
	    IperpfilterName[6],  IperpfilterName[7],  IperpfilterName[8],
	    IperpfilterName[9], IperpfilterName[10], IperpfilterName[11] );
   maxIperpfilterindex = nfilters - 1;
   if( maxIperpfilterindex > (MAXFILTERS - 1) )
     Quit("Too many filters in the Iperp table.");

   while( fgets( inputline, 200, in) != NULL ) {
      sscanf( inputline, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
	        &xT, &xlogg, 
                &xIperp[0],  &xIperp[1],  &xIperp[2], 
                &xIperp[3],  &xIperp[4],  &xIperp[5],
	        &xIperp[6],  &xIperp[7],  &xIperp[8],
	        &xIperp[9],  &xIperp[10], &xIperp[11]);
      gindex = (xlogg - gmin + 0.01) / deltag;
      if( (gindex < 0) || (gindex > maxIperpgindex) )
         Quit("gindex out of range in ReadIperpTable.");
      Tindex = ( xT - Tmin + 1.0) / deltaT;
      if( (Tindex < 0) || (Tindex > maxIperpTindex) )
         Quit("Tindex out of range in ReadIperpTable.");
      for( findex = 0; findex <= maxIperpfilterindex; findex++) {
         Iperptable[gindex][Tindex][findex] = xIperp[findex];
      }
   }
   fclose ( in );

   return;
}


void ReadIBBfilterTable( void )
/*****************************************************************
*
*   This function reads the intensities of a black body
*   observed through a filter.
*
*   The file name must be "IBBfilter.dat" and must have the format:
*       
*       Tmin    Tmax    deltaT    =  The minimum and maximum temperature
*                                    and the temperature spacing.
*         N filtername1 filtername2 filtername3 ... filternameN
*        T   Ifilter1  Ifilter2  Ifilter3 ... IfilterN
*        .      .         .         .            .
*        .      .         .         .            .
*        .      .         .         .            .
*
*   The filter can have comment lines beginning with a "*" at the
*   beginning.
*
*******************************************************************/
{
   FILE *in;
   char filename[40], inputline[201];
   long i, nfilters, Tindex, findex;
   double xT, xIBB[20];

   strcpy(filename, "IBBfilterTable.dat");
   if((in = fopen(filename, "r")) == NULL)
      Quit("Cannot open file IBBfilter.dat.");

   /******************************************************
   *
   *   Skip over any initial comment lines (lines beginning
   *   with a '*') then read two lines containing the
   *   information about the contents of the table.
   *
   ********************************************************/
   for(;;) {
      fgets(inputline, 80, in);
      if( inputline[0] != '*' ) {
            sscanf( inputline, "%lf %lf %lf", 
                                &IBBTmin, &IBBTmax, &IBBdeltaT);   
            maxIBBTindex = ( IBBTmax - IBBTmin + 0.1 ) / IBBdeltaT;
            for( i = 0; i <= maxIBBTindex; i++) {
               IBBT[i] = IBBTmin + i * IBBdeltaT;
            }
         fgets(inputline, 80, in);
            sscanf( inputline, "%ld %s %s %s %s %s %s %s %s %s %s %s %s", 
	           &nfilters,
                   IBBfilterName[0], IBBfilterName[1],  IBBfilterName[2],
                   IBBfilterName[3], IBBfilterName[4],  IBBfilterName[5],
                   IBBfilterName[6], IBBfilterName[7],  IBBfilterName[8],
	           IBBfilterName[9], IBBfilterName[10], IBBfilterName[11]);
            maxIBBfilterindex = nfilters - 1;
            if( maxIBBfilterindex > (MAXFILTERS - 1) )
	      Quit("Too many filters in the IBBfilter table.");
         break;
      }
   }
   while( fgets( inputline, 200, in) != NULL ) {
      sscanf( inputline, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
	        &xT,
                &xIBB[0], &xIBB[1], &xIBB[2], &xIBB[3], &xIBB[4],  &xIBB[5],
	        &xIBB[6], &xIBB[7], &xIBB[8], &xIBB[9], &xIBB[10], &xIBB[11]);
      Tindex = ( xT - IBBTmin + 1.0) / IBBdeltaT;
      if( (Tindex < 0) || (Tindex > maxIBBTindex) )
         Quit("Tindex out of range in ReadIBBTable.");
      for( findex = 0; findex <= maxIBBfilterindex; findex++) {
         IBBtable[Tindex][findex] = xIBB[findex];
      }
   }
   fclose ( in );

   return;
}


void ReadZzetaTable( void )
/*******************************************************
*
*   Read the ZBBzeta table
*
***********************************************************/
{
   FILE *in;
   char filename[40];
   long i;
   double dummy;

   strcpy( filename, "ZzetaTable.dat");
   if((in = fopen(filename, "r")) == NULL)
      Quit("Cannot open file ZzetaTable.dat.");

   fscanf( in, "%ld %lf", &maxBBzetaindex, &deltaBBzeta);
   for( i = 0; i <= maxBBzetaindex; i++) {
     fscanf( in, "%lf %lf", &dummy, &ZBBzeta[i]);
   }
   BBzetamax = maxBBzetaindex * deltaBBzeta;

   fclose( in );

   return;
}
