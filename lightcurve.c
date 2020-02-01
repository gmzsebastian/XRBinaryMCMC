/******************************************************
*
*                    FILE LIGHTCURVE.C
*
*   Modified 01/13/2005:  Fixed bugs in the calculation of
*   T2I[][] and TDiskI[][].
*
******************************************************/

#include "header.h"

void MakeLightCurves( void )
/***************************************************************
*
*   This function makes the orbital light curve.
*
*******************************************************/
{
   long iphase, k, band;
   double TotalFlux[MAXBANDPASSES];
   if( strcmp( control.thirdlight, "ON" ) == 0 ) {
      if( strcmp( verbose, "ONX" ) == 0 ) {
         printf(" Calculating third light fluxes.\n");
      }
      ThirdLight();
   }
   if( strcmp( verbose, "ONX" ) == 0 ) 
      printf(" Begin calculating the light curves.\n");

   for( iphase = 0; iphase <= orbit.maxpindex; iphase++ ) {
      LCphase[iphase] = orbit.phasemin + iphase * orbit.deltaphase;
      k = iphase / 10;
      k = iphase - 10 * k;
      if( k == 0 ) {
         if( strcmp( verbose, "ONX" ) == 0 ) {
	    printf("    phase number %3ld   phase =  %7.4f\n", iphase, 
                                                         LCphase[iphase]);
         }
      }
      FluxesAtPhase( LCphase[iphase], TotalFlux );
      for( band = 1; band <= orbit.nbands; band++) {
         LCflux[band][iphase] = TotalFlux[band];
      }
   }

   if( strcmp( orbit.normalize, "OFF" ) != 0 ) {
      if( strcmp( verbose, "ONX" ) == 0 ) {
	 printf(" Normalizing the light curves.\n");
      }
      Normalize();
   }
   else {
      for( band = 1; band <= orbit.nbands; band++) {
         for( iphase = 0; iphase <= orbit.maxpindex; iphase++ ) {
	    NormLC[band][iphase] = 0.0;
         }
      }
   }

   return;
}

void FluxesAtPhase( double phase, double TotalFlux[] )
/**************************************************
*
*   This function calculates the total fluxes from the system at
*   a particular orbital phase.  It is the heart of the program.
*
*******************************************************/
{
   struct CartVector sunvector, start;
   long itile, k, band;
   double      T2mu[MAX2TILES],     TDiskmu[MAXDISKTILES],
                  Star1Emitted,               Star1Escape,
          T2Emitted[MAX2TILES],       T2Escape[MAX2TILES],  
    TDiskEmitted[MAXDISKTILES], TDiskEscape[MAXDISKTILES],
              InnerDiskEmitted,           InnerDiskEscape;
   double x, calcphase, lambda, star1flux, star2flux, diskflux, 
          InnerDiskflux, emittedflux, InnerDiskMu;

   calcphase = phase - orbit.phaseoffset;
   if( calcphase < -0.5 ) 
      calcphase += 1.0;
   if( calcphase > 1.0 )
      calcphase -= 1.0;
   sunvector.x = -1.0 * sin( syspars.i ) * sin( calcphase * TWOPI );
   sunvector.y =        cos( syspars.i );
   sunvector.z = -1.0 * sin( syspars.i ) * cos( calcphase * TWOPI );

   /**********************************************************
   *
   *   The code first calculates the quanties 
   *        T2mu[], TDiskmu[],
   *        Star1Escape, T2Escape[], TDiskEscape[], 
   *          and InnerDiskEscape.
   *   These are the fractional amounts of the light emitted
   *   from each object or tile that escapes from the system.
   *   In the current version this fraction is a function only of 
   *   geometry, and so is the the same for all bandpasses.
   *
   ***********************************************************/
   if( strcmp( control.star1, "ON" ) == 0 ) {
      start.x = 0.0;
      start.y = 0.0;
      start.z = syspars.a;
         Star1Escape = EscapeFraction( start, sunvector);
   }

   if( strcmp( control.star2, "ON" ) == 0 ) {
      for( itile = 1; itile <= star2.Ntiles; itile++) {
         T2mu[itile] = CartDotProd( sunvector, T2normCart[itile] );
         if( T2mu[itile] > 0.0 ) {
	    start.x = T2x[itile];
            start.y = T2y[itile];
            start.z = T2z[itile];
            T2Escape[itile] = EscapeFraction( start, sunvector);
         }
         else{
            T2Escape[itile] = 0.0;
         }
      }
   }

   if( strcmp( control.disk, "ON" ) == 0 ) {
      for( itile = 1; itile <= disk.Ntiles; itile++) {
	 TDiskmu[itile] = CartDotProd( sunvector, TDisknormCart[itile] );
         if( TDiskmu[itile] > 0.0 ) {
            start.x = TDiskx[itile];
            start.y = TDisky[itile];
            start.z = TDiskz[itile];
            TDiskEscape[itile] = EscapeFraction( start, sunvector);
         }
         else{
            TDiskEscape[itile] = 0.0;
         }
      }
      if( strcmp( control.innerdisk, "ON" ) == 0 ) {
         start.x = 0.0;
         start.y = 0.0;
         start.z = syspars.a;
            InnerDiskEscape = EscapeFraction( start, sunvector);
      }
   }

   /**********************************************************
   *
   *   Now calculate the fluxes and then the fraction of the
   *   flux that escapes and add to the light curve.  This is
   *   done for each bandpass.
   *
   **********************************************************/
   for( band = 1; band <= orbit.nbands; band++) {
      TotalFlux[band] = 0.0;

      /******************************************************
      *
      *   Add the flux from star 1 if it is ON.
      *
      *******************************************************/
      if( strcmp( control.star1, "ON" ) == 0 ) {
	 Star1Emitted = Star1Flambda( orbit.filter[band], 
                                   orbit.minlambda[band],
                                   orbit.maxlambda[band] );
         star1flux = Star1Emitted * Star1Escape;
         TotalFlux[band] += star1flux;
      }

      /******************************************************
      *
      *   Add the flux from star 2 if it is ON.
      *
      *******************************************************/
      if( strcmp( control.star2, "ON" ) == 0 ) {
         star2flux = 0.0;
         if( strcmp( orbit.filter[band], "SQUARE") == 0 ) {
            for( itile = 1; itile <= star2.Ntiles; itile++) {
	       if( T2mu[itile] < 0.0 ) {
	          T2Emitted[itile] = 0.0;
               }
               else {
   	          T2Emitted[itile] =  T2I[band][itile] 
                                         * T2mu[itile] * T2dS[itile];
/*************************************
*
*   The following code fragment is useful for testing this part of the code.
*
*   if( itile == 28900 ) {
*     printf("  itile = %5d  T2I = %12.4e  T2mu = %8.5f  T2dS = %12.4e\n",
*    	            itile, T2I[band][itile], T2mu[itile], T2dS[itile]);
*   }
*
******************************************/
               }
               star2flux += T2Emitted[itile] * T2Escape[itile];
            }
         }
         else {
            for( itile = 1; itile <= star2.Ntiles; itile++) {
	       if( T2mu[itile] < 0.0 ) {
		  T2Emitted[itile] = 0.0;
               }
	       else {
	          if( (T2T[itile] > IperpT[maxIperpTindex]) 
                                        || (T2T[itile] < IperpT[0]) ) {
      	             T2Emitted[itile] = T2I[band][itile] 
                                         * T2mu[itile] * T2dS[itile];
                  }
                  else {
                     T2Emitted[itile] = T2I[band][itile]
                                        * ClaretHmu( T2T[itile], 
                                                    T2logg[itile],
                                                     orbit.filter[band],
                                                     T2mu[itile] )
                                        * T2mu[itile] * T2dS[itile];
/*************************************
*
*   The following code fragment is useful for testing ClaretHmu().
*
*   if( itile == 28900 ) {
*     x = ClaretHmu( T2T[itile], T2logg[itile],orbit.filter[band],
*                                            T2mu[itile] )
*     printf("  itile = %5d   T2mu = %8.5f  Hmu = %12.4e\n",
*    	            itile, T2mu[itile], x);
*   }
*
******************************************/
                  }
               }
               star2flux += T2Emitted[itile] * T2Escape[itile];
            }
         }
         TotalFlux[band] += star2flux;
      }

      /******************************************************
      *
      *   Add the flux from the disk if it is ON.  Note that
      *   the effects of the filter are already included in 
      *   TDiskI[band][].
      *
      *******************************************************/
      if( strcmp( control.disk, "ON" ) == 0 ) {
	 diskflux = 0.0;
         for( itile = 1; itile <= disk.Ntiles; itile++) {
            if( TDiskmu[itile] < 0.0 ) {
	       TDiskEmitted[itile] = 0.0;
	    }
            else {
   	       TDiskEmitted[itile] = TDiskI[band][itile] 
                                      * TDiskmu[itile] * TDiskdS[itile];
            }
            diskflux += TDiskEmitted[itile] * TDiskEscape[itile];
         }
         TotalFlux[band] += diskflux;
         if( strcmp( control.innerdisk, "ON" ) == 0 ) {
	    InnerDiskMu = sunvector.y;
	    InnerDiskEmitted = InnerDiskFlambda( orbit.filter[band], 
                                              orbit.minlambda[band],
                                             orbit.maxlambda[band] );
            InnerDiskflux = InnerDiskMu * InnerDiskEmitted * InnerDiskEscape;
            TotalFlux[band] += InnerDiskflux;
         }
      }

      /***************************************************
      *
      *   Add the third light flux if THIRDLIGHT is ON
      *
      *****************************************************/
      if( strcmp( control.thirdlight, "ON" ) == 0 ) {
         TotalFlux[band] += thirdlight.addFlux[band];
      }

      if( strcmp( control.diagnostics, "INSPECTESCAPE") == 0 ) {
	 if( strcmp( control.diagnoseband, orbit.filter[band] ) == 0 ) {
            InspectEscape( sunvector, orbit.filter[band],
                           Star1Emitted, Star1Escape,
                           T2mu, T2Emitted, T2Escape,
                           TDiskmu, TDiskEmitted, TDiskEscape);
            Quit("Quit after INSPECTESCAPE.");
	 }
      }
   }

   return;
}


void ThirdLight( void )
/**************************************************
*
*   This function calculates the amount of third light flux to
*   to be added to each bandpass.  The flux is actually added in
*   function FluxesAtPhase().  
*
*******************************************************/
{
   long LCband, found, band, iphase;
   double TotalFlux[MAXBANDPASSES], alpha, I3;

   FluxesAtPhase( thirdlight.orbphase, TotalFlux );

   for( LCband = 1; LCband <= orbit.nbands; LCband++) {
      found = 0;
      for( band = 1; band <= thirdlight.nbands; band++) {
	 if( strcmp( orbit.filter[LCband], thirdlight.filter[band]) == 0 ) {
	    if( strcmp( orbit.filter[LCband], "SQUARE") == 0 ) {
               if( (orbit.minlambda[LCband] == thirdlight.minlambda[band]) &&
		   (orbit.maxlambda[LCband] == thirdlight.maxlambda[band]) ) {
		  found = 1;
                  alpha = thirdlight.fraction[band];
               }
            }
            else {
	       found = 1;
               alpha = thirdlight.fraction[band];
            }
         }
      }
      if( found == 1 ) {
	 thirdlight.addFlux[LCband] = ( alpha / (1.0 - alpha) )
                                                     * TotalFlux[LCband];
      }
      else {
         Quit("Thirdlight fraction not specified for a bandpass.");
      }
   }

   return;
}


void Normalize( void )
/**************************************************
*
*   This function normalizes the orbital light.  Meaning of the
*   normalization types:
*       MAXVALUE  normvalue
*            The maximum value of each individual light curve is 
*            normalized to normvalue.
*       FITDATA  normfilter  minlambda maxlambda
*            The normalization factor for filter "normfilter" is set
*            equal to the value that minimizes the (weighted) mean squared
*            squared difference between the observed light curve
*            and the calculated light curve.  This same normalization
*            factor is then applied to all the calculated light curves.
*
*******************************************************/
{
   long iphase, band, calcband, databand, i, k;
   double maxflux, normfactor, sum1, sum2, weight,
          TotalFlux[MAXBANDPASSES], error;

   if( strcmp( orbit.normalize, "MAXVALUE") == 0 ) {
      for( band = 1; band <= orbit.nbands; band++) {
         maxflux = 0.0;
         for( iphase = 0; iphase <= orbit.maxpindex; iphase++) {
            if( LCflux[band][iphase] > maxflux )
               maxflux = LCflux[band][iphase];
         }
         if( maxflux == 0.0 )
            maxflux = 1.0;
         normfactor = orbit.normvalue / maxflux;
         for( iphase = 0; iphase <= orbit.maxpindex; iphase++ ) {
            NormLC[band][iphase] = normfactor * LCflux[band][iphase];
         }
      }
   }
   else if( strcmp( orbit.normalize, "FITDATA") == 0 ) {
      calcband = 0;
      for( band = 1; band <= orbit.nbands; band++){
         if( strcmp( orbit.normfilter, orbit.filter[band]) == 0 ) {
            if( strcmp( orbit.normfilter, "SQUARE") == 0 ) {
               if( (orbit.normMinlambda == orbit.minlambda[band]) &&
                   (orbit.normMaxlambda == orbit.maxlambda[band]) ) {
       	          calcband = band;
               }
            }
            else {
               calcband = band;
            }
         }
      }               
      if( calcband == 0 )
         Quit("Could not find a matching calculated band in Normalize().");
                       
      databand = 0;
      for( band = 1; band <= data.nbands; band++){
         if( strcmp( orbit.normfilter, data.filter[band]) == 0 ) {
            if( strcmp( orbit.normfilter, "SQUARE") == 0 ) {
               if( (orbit.normMinlambda == data.minlambda[band]) &&
                   (orbit.normMaxlambda == data.maxlambda[band]) ) {
                  databand = band;
               }
            }
            else {
               databand = band;
            }
         }
      }
      if( databand == 0 ) {
         Quit("Could not find a matching data band in Normalize().");
      }

      if( strcmp( verbose, "ONX") == 0 ) {
	 if( strcmp( orbit.normfilter, "SQUARE") == 0 ) {
	    printf("    Normalizing in the SQUARE %7.1f %7.1f bandpass.\n",
                                  orbit.normMinlambda, orbit.normMaxlambda);
         }
         else {
	    printf("    Normalizing in the %s filter.\n", orbit.normfilter);
         }
      }
             
      sum1 = 0.0;
      sum2 = 0.0;
      for( i = 1; i <= data.npoints[databand]; i++) {
         k = i / 10;
         k = i - 10 * k;
         if( k == 0 ) {
            if( strcmp( verbose, "ONX" ) == 0 ) {
               printf("    phase number %3ld   phase =  %7.4f\n",
                                     i, data.phase[databand][i]);
            }
         }
         FluxesAtPhase( data.phase[databand][i], TotalFlux );
         weight = 1.0 / ( data.standdev[databand][i] 
                       * data.standdev[databand][i] );
         sum1 += weight * data.flux[databand][i] * TotalFlux[calcband];
         sum2 += weight * TotalFlux[calcband] * TotalFlux[calcband];
      }
      normfactor = sum1 / sum2;
      for( band = 1; band <= orbit.nbands; band++) {
         for( iphase = 0; iphase <= orbit.maxpindex; iphase++ ) {
            NormLC[band][iphase] = normfactor * LCflux[band][iphase];
         }
      }

      data.chisquare = 0.0;
      for( i = 1; i <= data.npoints[databand]; i++) {
	 FluxesAtPhase( data.phase[databand][i], TotalFlux );
         weight = 1.0 / ( data.standdev[databand][i] 
                       * data.standdev[databand][i] );
         error = data.flux[databand][i] - normfactor * TotalFlux[calcband];
         data.chisquare += - 0.5 * (weight * error * error + log(2 * M_PI / weight));
      if( strcmp( verbose, "ON") == 0 ) 
   printf("S %7.5f T %11.6e ", data.phase[databand][i], normfactor * TotalFlux[calcband]);
      }
   }
   else {
      Quit("Unrecognized normalization type in function Normalize().");
   }

   return;
}


void Irradiate( void )
/***************************************************************
*
*   This function calculates the heating due to irradiation.  Note:
*      -- Disk irradiation is calculated first and includes 
*            irradiation from star 1, the ADC, and the inner disk.
*      -- Star 2 irradiation is calculated second and includes
*            irradiation from star 1, the ADC, the inner disk, and the 
*            (possibly heated) disk.
*
*   In the current version the disk does not heat itself except
*   for the inner disk.
*
*   Also in the current version the ADC acts as if all its flux
*   comes from two points located at y = +/1 adc.height above
*   and below the disk; 1/2 of adc.L from each point.
*
*   The code sacrificies efficiency for clarity in several places.
*
*****************************************************************/
{
   struct CartVector start, end, delta, vectord, direction;
   long iDisktile, i2tile, k, band;
   double d, dsquare, muEprime, muE, muAprime, muA, sum, 
          DeltaT4star1, DeltaT4disk, delT4disk, DeltaT4, x, y;
   double     TDiskTold[MAXDISKTILES],        muA1toD[MAXDISKTILES],
           transmit1toD[MAXDISKTILES],    DeltaT41toD[MAXDISKTILES],
               muAidtoD[MAXDISKTILES],  transmitidtoD[MAXDISKTILES],
           DeltaT4idtoD[MAXDISKTILES],      muAADCtoD[MAXDISKTILES],
          DeltaT4ADCtoD[MAXDISKTILES], transmitADCtoD[MAXDISKTILES];
   double         T2Told[MAX2TILES],          muA1to2[MAX2TILES], 
            transmit1to2[MAX2TILES],      DeltaT41to2[MAX2TILES], 
                muAidto2[MAX2TILES], transmitidto2[MAXDISKTILES],
            DeltaT4idto2[MAX2TILES],      DeltaT4Dto2[MAX2TILES],
               muAADCto2[MAX2TILES],    DeltaT4ADCto2[MAX2TILES],
          transmitADCto2[MAX2TILES];

   if( strcmp( verbose, "ONX" ) == 0 ) 
      printf(" Begin heating by irradiation.\n");
   /*********************************************************
   *
   *   If the disk is "ON", heat the disk by flux from star 1, the
   *   inner disk, and the ADC.  All these are treated in much the
   *   same way except:
   *         Star 1:  Point source at position of star 1
   *     Inner Disk:  Point source at position of star 1 but
   *                  suffers cos( theta ) geometric foreshortening.
   *            ADC:  Two point sources prependicular above and
   *                  below the position of star 1.
   *
   ************************************************************/
   if( strcmp( control.disk, "ON" ) == 0) {

      /************************************************
      *  
      *   Heat the outer disk by star 1.
      *
      ***********************************************/
      if (strcmp( control.star1, "ON") == 0 ) {
         if( strcmp( verbose, "ONX" ) == 0 ) 
            printf("   Begin heating the outer disk by star 1.\n");

         for( iDisktile = 1; iDisktile <= disk.Ntiles; iDisktile++) {
            TDiskTold[iDisktile] = TDiskT[iDisktile];
         }
         start.x = 0.0;
         start.y = 0.0;
         start.z = syspars.a;
         for( iDisktile = 1; iDisktile <= disk.Ntiles; iDisktile++) {
            end.x = TDiskx[iDisktile];
            end.y = TDisky[iDisktile];
            end.z = TDiskz[iDisktile];
            delta.x = end.x - start.x;
            delta.y = end.y - start.y;
            delta.z = end.z - start.z;
            d =  sqrt( delta.x*delta.x + delta.y*delta.y + delta.z*delta.z );
            direction.x = delta.x / d;
            direction.y = delta.y / d;
            direction.z = delta.z / d;
            muA1toD[iDisktile] = CartDotProd( direction, 
                                               TDisknormCart[iDisktile]);
            if( muA1toD[iDisktile] < 0.0 ) {
               DeltaT41toD[iDisktile] = fabs( Star1TotFlux( d )
                                            * muA1toD[iDisktile]
                                            / SIGMA );
               transmit1toD[iDisktile] = Transmission( start, end );
            }
            else {
               DeltaT41toD[iDisktile] = 0.0;
               transmit1toD[iDisktile] = 0.0;
            }
            TDiskT4[iDisktile] = TDiskT4[iDisktile]
                                 + disk.albedo * DeltaT41toD[iDisktile] 
                                 * transmit1toD[iDisktile];
            TDiskT[iDisktile] = pow( TDiskT4[iDisktile], 0.25 );
         }
         if( strcmp( control.diagnostics, "INSPECTHEATING") == 0 ) {
            InspectHeatDiskBy1(TDiskTold, muA1toD, DeltaT41toD, transmit1toD );
         }
      }

      /*********************************************************
      *
      *   Heat the outer disk by the inner disk.
      *
      ************************************************************/
      if (strcmp( control.innerdisk, "ON") == 0 ) {
         if( strcmp( verbose, "ONX" ) == 0 ) 
            printf("   Begin heating the outer disk by the inner disk.\n");

         for( iDisktile = 1; iDisktile <= disk.Ntiles; iDisktile++) {
            TDiskTold[iDisktile] = TDiskT[iDisktile];
         }
         start.x = 0.0;
         start.y = 0.0;
         start.z = syspars.a;
         for( iDisktile = 1; iDisktile <= disk.Ntiles; iDisktile++) {
            end.x = TDiskx[iDisktile];
            end.y = TDisky[iDisktile];
            end.z = TDiskz[iDisktile];
            delta.x = end.x - start.x;
            delta.y = end.y - start.y;
            delta.z = end.z - start.z;
            d =  sqrt( delta.x*delta.x + delta.y*delta.y + delta.z*delta.z );
            direction.x = delta.x / d;
            direction.y = delta.y / d;
            direction.z = delta.z / d;
            muAidtoD[iDisktile] = CartDotProd( direction, 
                                               TDisknormCart[iDisktile]);
            if( muAidtoD[iDisktile] < 0.0 ) {
               DeltaT4idtoD[iDisktile] = fabs( InnerDiskTotFlux( d ) 
                                         * direction.y
                                         * muAidtoD[iDisktile] 
                                         / SIGMA );
               transmitidtoD[iDisktile] = Transmission( start, end );
            }
            else {
               DeltaT4idtoD[iDisktile] = 0.0;
               transmitidtoD[iDisktile] = 0.0;
            }
            TDiskT4[iDisktile] = TDiskT4[iDisktile]
                                 + disk.albedo * DeltaT4idtoD[iDisktile] 
                                 * transmitidtoD[iDisktile];
            TDiskT[iDisktile] = pow( TDiskT4[iDisktile], 0.25 );
         }
         if( strcmp( control.diagnostics, "INSPECTHEATING") == 0 ) {
            InspectHeatDiskByID(TDiskTold, muAidtoD, DeltaT4idtoD, 
                                   transmitidtoD );
         }
      }

      /************************************************
      *  
      *   Heat the outer disk by the ADC.  Note that the ADC is
      *   represented by two points, one above and one below the
      *   disk.
      *
      ***********************************************/
      if (strcmp( control.adc, "ON") == 0 ) {
         if( strcmp( verbose, "ONX" ) == 0 ) 
            printf("   Begin heating the outer disk by the ADC.\n");

         for( iDisktile = 1; iDisktile <= disk.Ntiles; iDisktile++) {
            TDiskTold[iDisktile] = TDiskT[iDisktile];
         }
         /*********************************
	 *  The upper ADC point.
	 **********************************/
         start.x = 0.0;
         start.y = adc.height;
         start.z = syspars.a;
         for( iDisktile = 1; iDisktile <= disk.Ntiles; iDisktile++) {
            end.x = TDiskx[iDisktile];
            end.y = TDisky[iDisktile];
            end.z = TDiskz[iDisktile];
            delta.x = end.x - start.x;
            delta.y = end.y - start.y;
            delta.z = end.z - start.z;
            d =  sqrt( delta.x*delta.x + delta.y*delta.y + delta.z*delta.z );
            direction.x = delta.x / d;
            direction.y = delta.y / d;
            direction.z = delta.z / d;
            muAADCtoD[iDisktile] = CartDotProd( direction, 
                                               TDisknormCart[iDisktile]);
            if( muAADCtoD[iDisktile] < 0.0 ) {
               DeltaT4ADCtoD[iDisktile] = fabs( ADCTotFlux( d ) 
                                         * muAADCtoD[iDisktile] 
                                         / SIGMA );
               transmitADCtoD[iDisktile] = Transmission( start, end );
            }
            else {
               DeltaT4ADCtoD[iDisktile] = 0.0;
               transmitADCtoD[iDisktile] = 0.0;
            }
            TDiskT4[iDisktile] = TDiskT4[iDisktile]
                                 + disk.albedo * DeltaT4ADCtoD[iDisktile] 
                                 * transmitADCtoD[iDisktile];
            TDiskT[iDisktile] = pow( TDiskT4[iDisktile], 0.25 );
         }
         if( strcmp( control.diagnostics, "INSPECTHEATING") == 0 ) {
            InspectHeatDiskByADC( "TOP", TDiskTold, muAADCtoD, 
                                  DeltaT4ADCtoD,  transmitADCtoD );
         }
         /*********************************
	 *  The lower ADC point.
	 **********************************/
         start.x = 0.0;
         start.y = -adc.height;
         start.z = syspars.a;
         for( iDisktile = 1; iDisktile <= disk.Ntiles; iDisktile++) {
            end.x = TDiskx[iDisktile];
            end.y = TDisky[iDisktile];
            end.z = TDiskz[iDisktile];
            delta.x = end.x - start.x;
            delta.y = end.y - start.y;
            delta.z = end.z - start.z;
            d =  sqrt( delta.x*delta.x + delta.y*delta.y + delta.z*delta.z );
            direction.x = delta.x / d;
            direction.y = delta.y / d;
            direction.z = delta.z / d;
            muAADCtoD[iDisktile] = CartDotProd( direction, 
                                               TDisknormCart[iDisktile]);
            if( muAADCtoD[iDisktile] < 0.0 ) {
               DeltaT4ADCtoD[iDisktile] = fabs( ADCTotFlux( d ) 
                                         * muAADCtoD[iDisktile] 
                                         / SIGMA );
               transmitADCtoD[iDisktile] = Transmission( start, end );
            }
            else {
               DeltaT4ADCtoD[iDisktile] = 0.0;
               transmitADCtoD[iDisktile] = 0.0;
            }
            TDiskT4[iDisktile] = TDiskT4[iDisktile]
                                 + disk.albedo * DeltaT4ADCtoD[iDisktile] 
                                 * transmitADCtoD[iDisktile];
            TDiskT[iDisktile] = pow( TDiskT4[iDisktile], 0.25 );
         }
         if( strcmp( control.diagnostics, "INSPECTHEATING") == 0 ) {
            InspectHeatDiskByADC( "BOTTOM", TDiskTold, muAADCtoD, 
                                  DeltaT4ADCtoD,  transmitADCtoD );
         }
      }

      /************************************************************
      *
      *   Calculate the integrated specific intensity emitted by each 
      *   heated disk tile.  The disk is assumed to emit angle-independent
      *   black body intensities. 
      *
      *************************************************************/
      for( band = 1; band <= orbit.nbands; band++) {
         if( strcmp( orbit.filter[band], "SQUARE") == 0 ) {
            for( iDisktile = 1; iDisktile <= disk.Ntiles; iDisktile++) {
               TDiskI[band][iDisktile] = BBSquareIntensity( TDiskT[iDisktile], 
                                     orbit.minlambda[band],
                                     orbit.maxlambda[band]);
            }
         }
         else {
            for( iDisktile = 1; iDisktile <= disk.Ntiles; iDisktile++) {
               TDiskI[band][iDisktile] = BBFilterIntensity( 
                                     TDiskT[iDisktile], orbit.filter[band]);
            }
         }
      }
   }

   /*********************************************************
   *
   *   If star 2 is "ON", heat the star 2 by flux from star 1, the
   *   inner disk, and the ADC.  All these are treated in much the
   *   same way except:
   *         Star 1:  Point source at position of star 1
   *     Inner Disk:  Point source at position of star 1 but
   *                  suffers cos( theta ) geometric foreshortening.
   *            ADC:  Two point sources prependicular above and
   *                  below the position of star 1.
   *   Also heat by the outer disk.
   *
   ************************************************************/
   if( strcmp( control.star2, "ON" ) == 0 ) {

      /*********************************************************
      *
      *   Heat star 2 by star 1
      *
      **********************************************************/
      if( strcmp( control.star1, "ON" ) == 0 ) {
         if( strcmp( verbose, "ONX" ) == 0 ) 
	    printf("   Begin heating star 2 by star 1.\n");

         for( i2tile = 1; i2tile <= star2.Ntiles; i2tile++) {
            T2Told[i2tile] = T2T[i2tile];
         }
         start.x = 0.0;
         start.y = 0.0;
         start.z = syspars.a;
         for( i2tile = 1; i2tile <= star2.Ntiles; i2tile++) {
	    end.x = T2x[i2tile];
            end.y = T2y[i2tile];
            end.z = T2z[i2tile];
            delta.x = end.x - start.x;
            delta.y = end.y - start.y;
            delta.z = end.z - start.z;
            d =  sqrt( delta.x*delta.x + delta.y*delta.y + delta.z*delta.z );
            direction.x = delta.x / d;
            direction.y = delta.y / d;
            direction.z = delta.z / d;
            muA1to2[i2tile] = CartDotProd( direction, T2normCart[i2tile] );
            if( muA1to2[i2tile] < 0.0 ) {
	       DeltaT41to2[i2tile] = fabs( Star1TotFlux( d )
                                           * muA1to2[i2tile] 
                                           / SIGMA );
               transmit1to2[i2tile] = Transmission( start, end);
            }
            else {
	       DeltaT41to2[i2tile] = 0.0;
	       transmit1to2[i2tile] = 0.0;
            }
            sum = pow( T2T[i2tile], 4.0 ) 
                   + star2.albedo * DeltaT41to2[i2tile] * transmit1to2[i2tile];
            T2T[i2tile] = pow( sum, 0.25 );
         }

         if( strcmp( control.diagnostics, "INSPECTHEATING") == 0 ) {
            InspectHeat2By1(T2Told, muA1to2, DeltaT41to2, transmit1to2 );
         }
      }
 
      if( strcmp( control.disk, "ON" )  == 0 ) {
         /*********************************************************
         *
         *   Heat star 2 by the inner disk
         *
         **********************************************************/
         if( strcmp( control.innerdisk, "ON") == 0 ) {
            if( strcmp( verbose, "ONX" ) == 0 ) 
               printf("   Begin heating star 2 by the inner disk.\n");
              
            for( i2tile = 1; i2tile <= star2.Ntiles; i2tile++) {
               T2Told[i2tile] = T2T[i2tile];
            }
            start.x = 0.0;
            start.y = 0.0;
            start.z = syspars.a;
            for( i2tile = 1; i2tile <= star2.Ntiles; i2tile++) {
               end.x = T2x[i2tile];
               end.y = T2y[i2tile];
               end.z = T2z[i2tile];
               delta.x = end.x - start.x;
               delta.y = end.y - start.y;
               delta.z = end.z - start.z;
               d =  sqrt( delta.x*delta.x + delta.y*delta.y 
                                                   + delta.z*delta.z );
               direction.x = delta.x / d;
               direction.y = delta.y / d;
               direction.z = delta.z / d;
               muAidto2[i2tile] = CartDotProd( direction, T2normCart[i2tile] );
               if( muAidto2[i2tile] < 0.0 ) {
                  DeltaT4idto2[i2tile] = fabs( InnerDiskTotFlux( d )
                                               * direction.y
                                               * muAidto2[i2tile]
                                               / SIGMA );
                  transmitidto2[i2tile] = Transmission( start, end);
               }
               else {
	          DeltaT4idto2[i2tile] = 0.0;
	          transmitidto2[i2tile] = 0.0;
               }
               sum = pow( T2T[i2tile], 4.0 ) 
                      + star2.albedo * DeltaT4idto2[i2tile] 
                                * transmitidto2[i2tile];
               T2T[i2tile] = pow( sum, 0.25 );
            }
            if( strcmp( control.diagnostics, "INSPECTHEATING") == 0 ) {
               InspectHeat2ByID(T2Told, muAidto2, DeltaT4idto2, 
                                       transmitidto2 );
            }
         }

         /*********************************************************
         *
         *   Heat star 2 by the outer disk.   This short section of
         *   code is takes a huge amount of time to execute because
         *   the number of times the inner loop is executed is
         *          star2.Ntiles * disk.Ntiles ~ 10^4 * 2 x 10^4
         *                                     ~ 2 x 10^8
         *   and the inner loop calls the time-consuming function
         *   Transmission().
         *
         **********************************************************/
         if( strcmp( verbose, "ONX" ) == 0 ) 
	    printf("   Begin heating star 2 by the outer disk.\n");

         for( i2tile = 1; i2tile <= star2.Ntiles; i2tile++) {
            T2Told[i2tile] = T2T[i2tile];
         }
         for( i2tile = 1; i2tile <= star2.Ntiles; i2tile++) {
            if( strcmp( verbose, "ONX" ) == 0 ) {
               k = i2tile / 100;
               k = i2tile - 100 * k;
               if( k == 0 ) {
	          printf("      heating star 2 tile number %5ld\n", i2tile);
               }
            }
   	    DeltaT4Dto2[i2tile] = 0.0;
            end.x = T2x[i2tile];
            end.y = T2y[i2tile];
            end.z = T2z[i2tile];
            for( iDisktile = 1; iDisktile <= disk.Ntiles; iDisktile++) {
	       start.x = TDiskx[iDisktile];
               start.y = TDisky[iDisktile];
               start.z = TDiskz[iDisktile];
               vectord.x = end.x - start.x;
               vectord.y = end.y - start.y;
               vectord.z = end.z - start.z;
               muEprime = CartDotProd( vectord, TDisknormCart[iDisktile] );
               if( muEprime > 0.0 ) {
                  muAprime = CartDotProd( vectord, T2normCart[i2tile] );
                  if( muAprime < 0.0 ) {
                     dsquare =   vectord.x * vectord.x 
                               + vectord.y * vectord.y 
                               + vectord.z * vectord.z;
                     d = sqrt( dsquare );
                     muE = muEprime / d;
                     muA = muAprime / d;
                     delT4disk = -( TDiskT4[iDisktile] / PI)
	                            * (muE * muA / dsquare) 
                                    * TDiskdS[iDisktile];
                     delT4disk *= Transmission( start, end );
                     DeltaT4Dto2[i2tile] += delT4disk;
                  }
	       }
            }
            sum = pow( T2T[i2tile], 4.0) + star2.albedo * DeltaT4Dto2[i2tile];
            T2T[i2tile] = pow( sum, 0.25 );
         }
         if( strcmp( control.diagnostics, "INSPECTHEATING") == 0 ) {
            InspectHeat2ByDisk( T2Told, DeltaT4Dto2, T2T);
         }
      }

      /************************************************
      *  
      *   Heat star 2 by the ADC. 
      *
      ***********************************************/
      if (strcmp( control.adc, "ON") == 0 ) {
         if( strcmp( verbose, "ONX" ) == 0 ) 
            printf("   Begin heating star 2 by the ADC.\n");
         for( i2tile = 1; i2tile <= star2.Ntiles; i2tile++) {
            T2Told[i2tile] = T2T[i2tile];
         }
   
         /*********************************
         *  The upper ADC point.
         **********************************/
         start.x = 0.0;
         start.y = adc.height;
         start.z = syspars.a;
         for( i2tile = 1; i2tile <= star2.Ntiles; i2tile++) {
            end.x = T2x[i2tile];
            end.y = T2y[i2tile];
            end.z = T2z[i2tile];
            delta.x = end.x - start.x;
            delta.y = end.y - start.y;
            delta.z = end.z - start.z;
            d =  sqrt( delta.x*delta.x + delta.y*delta.y + delta.z*delta.z );
            direction.x = delta.x / d;
            direction.y = delta.y / d;
            direction.z = delta.z / d;
            muAADCto2[i2tile] = CartDotProd( direction, T2normCart[i2tile] );
            if( muAADCto2[i2tile] < 0.0 ) {
               DeltaT4ADCto2[i2tile] = fabs( ADCTotFlux( d ) 
                                         * muAADCto2[i2tile] 
                                         / SIGMA );
               transmitADCto2[i2tile] = Transmission( start, end );
            }
            else {
               DeltaT4ADCto2[i2tile] = 0.0;
               transmitADCto2[i2tile] = 0.0;
            }
            sum = pow( T2T[i2tile], 4.0)
                                 + star2.albedo * DeltaT4ADCto2[i2tile] 
                                 * transmitADCto2[i2tile];
            T2T[i2tile] = pow( sum, 0.25 );
         }
         if( strcmp( control.diagnostics, "INSPECTHEATING") == 0 ) {
            InspectHeat2ByADC( "TOP", T2Told, muAADCto2, 
                                  DeltaT4ADCto2,  transmitADCto2 );
         }

         /*********************************
         *  The lower ADC point.
         **********************************/
         start.x = 0.0;
         start.y = -adc.height;
         start.z = syspars.a;
         for( i2tile = 1; i2tile <= star2.Ntiles; i2tile++) {
            end.x = T2x[i2tile];
            end.y = T2y[i2tile];
            end.z = T2z[i2tile];
            delta.x = end.x - start.x;
            delta.y = end.y - start.y;
            delta.z = end.z - start.z;
            d =  sqrt( delta.x*delta.x + delta.y*delta.y + delta.z*delta.z );
            direction.x = delta.x / d;
            direction.y = delta.y / d;
            direction.z = delta.z / d;
            muAADCto2[i2tile] = CartDotProd( direction, T2normCart[i2tile] );
            if( muAADCto2[i2tile] < 0.0 ) {
               DeltaT4ADCto2[i2tile] = fabs( ADCTotFlux( d ) 
                                         * muAADCto2[i2tile] 
                                         / SIGMA );
               transmitADCto2[i2tile] = Transmission( start, end );
            }
            else {
               DeltaT4ADCto2[i2tile] = 0.0;
               transmitADCto2[i2tile] = 0.0;
            }
            sum = pow( T2T[i2tile], 4.0)
                                 + star2.albedo * DeltaT4ADCto2[i2tile] 
                                 * transmitADCto2[i2tile];
            T2T[i2tile] = pow( sum, 0.25 );
         }
         if( strcmp( control.diagnostics, "INSPECTHEATING") == 0 ) {
            InspectHeat2ByADC( "BOTTOM", T2Told, muAADCto2, 
                                  DeltaT4ADCto2,  transmitADCto2 );
         }
      }

      /*************************************************************
      *
      *   Calculate the integrated specific intensity emitted by each 
      *   of the tiles on star 2.  
      *   
      *   The meanings of the specific intensity depends on the
      *   bandpass, temperature, and spectrum type:
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
      for( band = 1; band <= orbit.nbands; band++) {
         if( strcmp( orbit.filter[band], "SQUARE") == 0 ) {
            for( i2tile = 1; i2tile <= star2.Ntiles; i2tile++) {
	       T2I[band][i2tile] = BBSquareIntensity( T2T[i2tile], 
                                                 orbit.minlambda[band], 
                                                 orbit.maxlambda[band]);
            }
         }
         else {
            for( i2tile = 1; i2tile <= star2.Ntiles; i2tile++) {
  	       if( (T2T[i2tile] > IperpT[maxIperpTindex]) 
                                        || (T2T[i2tile] < IperpT[0]) ) {
	          T2I[band][i2tile] = BBFilterIntensity( T2T[i2tile], 
                                                     orbit.filter[band]);
               }
               else {
                  T2I[band][i2tile] = GetIperp( T2T[i2tile], T2logg[i2tile], 
                                           orbit.filter[band]);
               }
            }
         }
      }
   }

   if( strcmp( control.diagnostics, "INSPECTHEATING") == 0 ) {
      Quit("Quit after INSPECTHEATING.");
   }

   return;
}


double Transmission( struct CartVector start, struct CartVector end)
/**************************************************
*
*   This function traces light rays between two points in the binary
*   system.  It returns the fraction of the intensity at the beginning
*   of the ray that survives to the end of the ray.  In this first
*   version the transmission is either 1.0 or 0.0.
*
*   Note that the function jumps the ray a few stepsizes when if
*   first begins to propagate the ray, and the ray is assumed to have 
*   arrived at the end of its path if it gets to within 3 pixels of
*   the end point.  This is reasonable since the long dimension of a
*   typical tile is several pixels long.  These measures ameliorate 
*   but do not entirely  eliminate the nastier effects of pixelization.
*
*   NOTE:  Program XRbinary spends the vast majority of its time
*          in this function.
*
*******************************************************/
{
   struct CartVector delta, direction, ray, deltaray;
   long nsteps, ix, iz, ixEnd, izEnd;
   double length, stepsize, transmit, InverseDeltax, InverseDeltaz;

   ixEnd = 1.5 + ( end.x - Grid.xmin ) / Grid.deltax;
   izEnd = 1.5 + ( end.z - Grid.zmin ) / Grid.deltaz;

   delta.x = end.x - start.x;
   delta.y = end.y - start.y;
   delta.z = end.z - start.z;
   length = sqrt( delta.x * delta.x + delta.y * delta.y + delta.z * delta.z );
   direction.x = delta.x / length;
   direction.y = delta.y / length;
   direction.z = delta.z / length;
   nsteps = length / Grid.deltal;
   if( nsteps <= 3 ) {
      transmit = 1.0;
      return( transmit );
   }
   stepsize = length / nsteps;
   deltaray.x = stepsize * direction.x;
   deltaray.y = stepsize * direction.y;
   deltaray.z = stepsize * direction.z;

   ray.x = start.x + 3.0 * deltaray.x;
   ray.y = start.y + 3.0 * deltaray.y;
   ray.z = start.z + 3.0 * deltaray.z;

   /*********************************************************
   *  
   *   This is the culprit folks!  The following for() loop is where
   *   XRbinary spends almost all its time.
   *
   ****************************************************/
   InverseDeltax = 1.0 / Grid.deltax;
   InverseDeltaz = 1.0 / Grid.deltaz;
   for(;;) {
      ray.x += deltaray.x;
      ray.y += deltaray.y;
      ray.z += deltaray.z;
      ix = 1.5 + ( ray.x - Grid.xmin ) * InverseDeltax;
      iz = 1.5 + ( ray.z - Grid.zmin ) * InverseDeltaz;
      if( abs( ixEnd - ix ) <= 3 ) {
         if( abs( izEnd - iz ) <= 3 ) {
	    transmit = 1.0;
	    break;
         }
      }
      if( ray.y < Grid.Topy[ix][iz] ) {
         if( ray.y > Grid.Bottomy[ix][iz] ) {
	    transmit = 0.0;
            break;
         }
      }
   }

   return( transmit );
}


double EscapeFraction( struct CartVector start, struct CartVector direction)
/***********************************************************
*
*   This function traces light rays through the binary system to
*   determine if the light ray escapes the system and contributes
*   to the light curve.  It also determines the fraction of the 
*   intensity at the beginning of the ray that survives to escape
*   from the binary.  In this version the escape fraction is 0 or 1.
*
*   Note that the calculation starts after the ray has already
*   traversed a few steplengths.  This avoids some of the
*   problems introduced by pixelization.
*
**********************************************************/
{
   struct CartVector ray, deltaray;
   long ix, iz;
   double steplength, transmit;

   transmit = 1.0;
   steplength = 0.8 * Grid.deltal;
   deltaray.x = steplength * direction.x;
   deltaray.y = steplength * direction.y;
   deltaray.z = steplength * direction.z;

   ray.x = start.x + 3.0 * deltaray.x;
   ray.y = start.y + 3.0 * deltaray.y;
   ray.z = start.z + 3.0 * deltaray.z;
   for(;;) {
      ray.x = ray.x + deltaray.x;
      if( ray.x >= Grid.xmax )
	 break;
      if( ray.x <= Grid.xmin )
         break;
      ray.y = ray.y + deltaray.y;
      if( ray.y >= Grid.ymax )
	 break;
      if( ray.y <= Grid.ymin )
	 break;
      ray.z = ray.z + deltaray.z;
      if( ray.z >= Grid.zmax )
	 break;
      if( ray.z <= Grid.zmin )
	 break;
      ix = 1.5 + ( ray.x - Grid.xmin ) / Grid.deltax;
      iz = 1.5 + ( ray.z - Grid.zmin ) / Grid.deltaz;
      if( ray.y < Grid.Topy[ix][iz] ) {
         if( ray.y > Grid.Bottomy[ix][iz] ) {
	    transmit = 0.0;
            return( transmit );
         }
      }
   }

   return( transmit );
}


void MakeYlimits( void )
/**************************************************
*
*   This function finds the surfaces Grid.Topy[i][j] and Grid.Bottomy[i][j]
*   defined by the highest and lowest values of y of the components 
*   in the system.  The function also finds maximum and minimum
*   values of x, y, z needed so the edges of the grid cover
*   the stars and disk.
*
*******************************************************/
{
   long ix, iz, itile;
   double x, z, rho, a, sinzeta, coszeta, zeta, margin;

   if( strcmp( verbose, "ONX") == 0 ) {
     printf(" Begin making Ylimits grid.\n");
   }

   margin = 0.04 * syspars.a;

   Grid.xmin = 0.0;
   Grid.xmax = 0.0;
   Grid.ymin = 0.0;
   Grid.ymax = 0.0;
   Grid.zmin = 0.0;
   Grid.zmax = 0.0;
   if(    (strcmp( control.star1,     "ON" ) == 0)
       || (strcmp( control.innerdisk, "ON" ) == 0)
       || (strcmp( control.adc,       "ON" ) == 0)  ) {
      Grid.zmin = syspars.a;
      Grid.zmax = syspars.a;
   }
   if( strcmp( control.adc,       "ON" ) == 0 ) {
      Grid.ymin = -adc.height;
      Grid.ymax =  adc.height;
   }

   if( strcmp( control.star2, "ON" ) == 0 ) {
      for( itile = 1; itile <= star2.Ntiles; itile++ ) {
         if( T2x[itile] < Grid.xmin ) Grid.xmin = T2x[itile];
         if( T2x[itile] > Grid.xmax ) Grid.xmax = T2x[itile];
         if( T2y[itile] < Grid.ymin ) Grid.ymin = T2y[itile];
         if( T2y[itile] > Grid.ymax ) Grid.ymax = T2y[itile];
         if( T2z[itile] < Grid.zmin ) Grid.zmin = T2z[itile];
         if( T2z[itile] > Grid.zmax ) Grid.zmax = T2z[itile];
      }
   }
   if( strcmp( control.disk, "ON" ) == 0 ) {
      for( itile = 1; itile <= disk.Ntiles; itile++ ) {
         if( TDiskx[itile] < Grid.xmin ) Grid.xmin = TDiskx[itile];
         if( TDiskx[itile] > Grid.xmax ) Grid.xmax = TDiskx[itile];
         if( TDisky[itile] < Grid.ymin ) Grid.ymin = TDisky[itile];
         if( TDisky[itile] > Grid.ymax ) Grid.ymax = TDisky[itile];
         if( TDiskz[itile] < Grid.zmin ) Grid.zmin = TDiskz[itile];
         if( TDiskz[itile] > Grid.zmax ) Grid.zmax = TDiskz[itile];
      }
   }

   Grid.xmin -= margin;
   Grid.xmax += margin;
   Grid.ymin -= margin;
   Grid.ymax += margin;
   Grid.zmin -= margin;
   Grid.zmax += margin;

   Grid.Nxtiles = GRIDXTILES - 1;
   Grid.Nztiles = GRIDZTILES - 1;
   Grid.deltax = (Grid.xmax - Grid.xmin) / ( Grid.Nxtiles - 1 );
   Grid.deltaz = (Grid.zmax - Grid.zmin) / ( Grid.Nztiles - 1 );
   if( (Grid.deltax <=0.0) || (Grid.deltaz <= 0.0) )
      Quit("Either Grid.deltax or Grid.deltaz equals zero.");
   Grid.deltal = sqrt( Grid.deltax*Grid.deltax + Grid.deltaz*Grid.deltaz);
  
   for( ix= 1; ix <= Grid.Nxtiles; ix++) {
      for( iz = 1; iz <= Grid.Nztiles; iz++) {
         Grid.Topy[ix][iz]    = 0.0;
         Grid.Bottomy[ix][iz] = 0.0;
  	 x = Grid.xmin + (ix - 1) * Grid.deltax;
         z = Grid.zmin + (iz - 1) * Grid.deltaz;
         if( z < syspars.rL1 ) {
            if( strcmp( control.star2, "ON" ) == 0 ) {
	       Grid.Topy[ix][iz] = Star2TopY( x, z );
	       Grid.Bottomy[ix][iz] = -Grid.Topy[ix][iz];
            }
         }
         else {
            if( strcmp( control.disk, "ON" ) == 0 ) {
               rho = sqrt( x*x + (z - syspars.a)*(z - syspars.a) );
               coszeta = (z - syspars.a) / rho;
               zeta = acos( coszeta );
               if( x < 0.0 )
	          zeta = TWOPI - zeta;
               a = RhoToA( rho, zeta);
               Grid.Topy[ix][iz] = DiskTopH( a, zeta);
               Grid.Bottomy[ix][iz] = DiskBottomH( a, zeta);
            }
         }
      }
   }
   if( strcmp( control.diagnostics, "INSPECTYLIMITS") == 0 ) {
      InspectYlimits();
      Quit("Quit in MakeYlimits after INSPECTYLIMITS.");
   }

   return;
}
