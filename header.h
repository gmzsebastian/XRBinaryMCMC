/*****************************************************
*
*                  FILE  HEADER.H
*
*   This is the header file for program "XRbinary".
*
******************************************************/

/*******************************************************
*
*  Preprocessor commands
*
******************************************************/
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define PI     3.1415926536
#define HALFPI 1.5707963268
#define TWOPI  6.2831853072
#define FOURPI 12.56637062
#define G      6.6742e-8
#define H      6.626076e-27
#define C      2.99792459e10
#define K      1.38066e-16
#define SIGMA  5.6704e-5
#define MSOL   1.9891e33

#define MAXFILTERS       12   /* Maximum number of filters in the tables */
#define MAXBANDPASSES    21   /* Maximum number of band passes */
#define MAX2TILES     40506   /* Maximum number of tiles on star 2  */
#define MAXDISKTILES  40506   /* Maximum number of tiles on the disk */
#define MAXPHASES       501   /* Maximum number of orbital phases  */
#define MAXLIMBDARK     101   /* Maximum number of mu entries in LD table */
#define GRIDXTILES      401   /* Maximum number x grid points  */
#define GRIDZTILES      401   /* Maximum number of z grid points */
#define MAXZETAPOINTS   101   /* Maximum number of zeta points */
#define MAXDATAPOINTS  1001   /* Max number of data points in a light curve */
#define TMAX         8000.0   /* Maximum temperature for normal stel. atmos.*/


/*****************************************************
*
*   Structure prototypes.
*
*****************************************************/

struct CartVector {
   double x;
   double y;
   double z;
};

struct CylVector {
   double rho;
   double zeta;
   double h;
};

struct SphereVector {
   double r;
   double theta;
   double phi;
};

struct filenames {
   char parfile[20];     /* Input file name (from command line)          */
   char syspars[20];     /* Output file name (constructed)               */
   char lightcurves[20]; /* Lightcurve file name (constructed)           */
};

struct flowcontrol {
   char star1[20];       /* "ON" if the model includes star 1            */
   char star2[20];       /* "ON" if the model includes star 2            */
   char star2spots[20];  /* "ON" if the model includes a spot on star 2  */
   char disk[20];        /* "ON" if the model includes a disk            */
   char diskrim[20];     /* "ON" if the model includes a disk rim        */
   char disktorus[20];   /* "ON" if the model includes a disk torus      */
   char innerdisk[20];   /* "ON" if the model includes a hot inner disk  */
   char diskspots[20];   /* "ON" if the model includes disk spots        */
   char adc[20];         /* "ON" if the model includes an ADC            */
   char thirdlight[20];  /* "ON" if the model includes a third light     */
   char irradiation[20]; /* "ON" if model includes irradiation heating   */ 
   char diagnostics[30]; /* "OFF" for no diagnostic output               */
   double diagnosephase; /* phase and bandpass for which to write        */
   char diagnoseband[30];/* diagnostic info. Used only in INSPECTESCAPE  */
   long diagnoseindex;   /* phase index at which to wrt diagnostic info  */
};

struct orbitparams {
   double phasemin;     /* minimum orbital phase of the light curve      */
   double phasemax;     /* maximum orbital phase of the light curve      */
   double deltaphase;   /* phase interval between light curve points     */
   long   maxpindex;    /* maximum index of light curve array            */
   double phaseoffset ; /* phase offset.  See input.txt for the sign.    */
   long nbands;         /* number of bandpasses used to make light curves*/
   char filter[MAXBANDPASSES][20];  
                        /* the filter name or SQUARE                     */
   double minlambda[MAXBANDPASSES]; 
                        /* min lambda for a square bandpass              */
   double maxlambda[MAXBANDPASSES]; 
                        /* max lambda for a square bandpass              */
   char normalize[20];  /* type of normalization                         */
   char normfilter[20]; /* filter to use when normalizing to data        */
   double normMinlambda;/* min lambda for square bandpass                */
   double normMaxlambda;/* max lambda for square bandpass                */
   double normvalue;    /* normalization value for NORMALIZE= MAXVALUE   */
};

struct systemparams {
   double p;            /*  orbital period                               */
   double omega;        /*  orbital angular frequency                    */
   double K2;           /*  amplitude of the radial velocity curve       */
   double q;            /*  mass ratio q = m2 / m1                       */
   double i;            /*  orbital inclination                          */
   double a;            /*  separation of the centers of mass            */
   double zcm;          /*  z coordinate of the center of mass           */
   double M1;           /*  mass of the compact star                     */
   double M2;           /*  mass of the secondary star                   */
   double rL1;          /*  distance of L1 from the center of star 2     */
   double VL1;          /*  potential at L1                              */
   double MeanLobe1Radius; /*  mean radius star 1 Roche lobe, units of a */
   double MeanLobe2Radius; /*  mean radius star 2 Roche lobe, units of a */
};

struct star1params {
   double L;            /*  luminosity of the compact star               */
   double T;            /*  effective  temperature of the compact star   */
   double sigmaT4;      /*  sigma * T^4                                  */
   double radius;       /*  radius of star 1 implied by its T and L      */
};

struct star2params {
   long targetNtiles;   /*  target number of tiles covering the star 2   */
   long Ntiles;         /*  number of tiles on the secondary's surface   */
   double volume;       /*  volume of the lobe-filling star              */
   double meanr;        /*  mean radius of the lobe-filling star         */
   double meang;        /*  mean surface gravity in cgs units            */
   double logg;         /*  log mean surface gravity in cgs units        */
   double meanT;        /*  Tmean, beta used in T = meanT(g/meang)^beta  */
   double beta;         /*  gravity darkening coefficient                */
   double albedo;       /*  albedo of the secondary star                 */
   double L;            /*  luminosity of star 2                         */
   double frontradius;  /*  radius of star 2 towards L1 (= syspars.rL1)  */
   double poleradius;   /*  radius at star 2 at its poles                */
   double sideradius;   /*  radius of star 2 at its sides                */
   double backradius;   /*  radius of star 2 opposite L1                 */
};

struct star2spotparams {
   long nspots;         /*  number of spots                              */
   double theta[20];    /*  spot theta position in radians               */
   double phi[20];      /*  spot phi position in radians                 */
   double radius[20];   /*  spot radius in radians                       */
   double SpotToverStarT[20];
                        /*  ratio of spot Teff to local star Teff        */
};

struct wholediskpars {
   long targetNtiles;   /*  target number of tiles to cover the disk     */
   long Ntiles;         /*  actual number of tiles                       */
   double e;            /*  disk eccentricity                            */
   double zetazero;     /*  disk periastron                              */
   double albedo;       /*  albedo of all disk structures                */
   double L;            /*  total luminosity of the disk                 */
   double TopTmax;         /*  highest temperature of the disk tiles        */
   double TopTmin;         /*  lowest temperature of the disk tiles         */
};

struct maindiskpars {
   double amin;         /*  minimum radius of the disk                   */
   double amax;         /*  maximum radius of the disk                   */
   double da;           /*  width of main disk tiles                     */
   double Hmax;         /*  maximum height of the main disk              */
   double Hpow;         /*  power in the disk height power law           */
   char Ttype[20];      /*  type of temperature distribution             */
   double maindiskL;    /*  luminosity of the main disk                  */
   double Tpow;         /*  power if the disk temperature is a power law */
   double TConstant;    /*  constant in the Teff equation                */
   double Tamax;        /*  temperature of main disk at its outer edge   */
   double Tamin;        /*  temperature of main disk at its inner edge   */
};

struct diskedgepars {
   double T;            /*  the temperature of the disk edge             */
   double Tspot;        /*  if Tspot > T, there is a spot on the edge    */
   double ZetaMid;      /*  zeta of the center of the spot               */
   double ZetaWidth;    /*  the full width of the spot                   */
};

struct diskrimpars {
   char type[20];       /*  model to use for the disk rim                */
   double awidth;       /*  width of the disk rim                        */
   double Hmax;         /*  maximum height of the disk rim               */
   double Hmin;         /*  minimum height of the disk rim               */
   double ZetaHmax;     /*  zeta where rim reaches its maximum height    */
   double Tmax;         /*  maximum temperature of the disk rim          */
   double Tmin;         /*  minimum temperature of the disk rim          */
   double ZetaTmax;     /*  zeta where the rim temperature is greatest   */
                        /*  The above six variables have different       */
                        /*    meanings for the SINUSOISAL and POINT rims!*/
   long points;         /*  number of points in the point-specified rim  */
   double PointZeta[MAXZETAPOINTS];
                        /*  zeta of a rim point                          */
   double PointH[MAXZETAPOINTS];
                        /*  height of a rim point                        */
   double PointT[MAXZETAPOINTS];
                        /*  temperature of a rim point                   */
}; 

struct disktoruspars {
   char type[20];       /*  model to use for the disk torus T and H      */
   double azero;        /*  distance of the torus from the disk center   */
   double awidth;       /*  full width of the torus                      */
   double Hmax;         /*  maximum height of the torus                  */
   double Hmin;         /*  minimum height of the torus                  */
   double ZetaHmax;     /*  zeta where torus reaches its maximum height  */
   double Tmax;         /*  maximum temperature of the disk torus        */
   double Tmin;         /*  minimum temperature of the disk torus        */
   double ZetaTmax;     /*  zeta where the torus temperature is greatest */
                        /*  The above six variables have different       */
                        /*    meanings for the SINUSOISAL and POINT rims!*/
   long points;         /*  number of points in the point-specified torus*/
   double PointZeta[MAXZETAPOINTS];
                        /*  zeta of a torus point                        */
   double PointH[MAXZETAPOINTS];
                        /*  height of a torus point                      */
   double PointT[MAXZETAPOINTS];
                        /*  temperature of a torus point                 */
};

struct diskspotpars {
   long nspots;           /*  Number of spots                            */
   double zetamin[20];    /*  Angle at the beginning of the spot         */
   double zetamax[20];    /*  Angle at the end of the spot               */
   double amin[20];       /*  a at the inner edge of the spot            */
   double amax[20];       /*  a at the outer edge of the spot            */
   double spotToverT[20]; /*  fractional change in the temperature       */
};

struct innerdiskpars {
   double T;            /*  temperature of the inner disk                */
   double L;            /*  luminosity of the inner disk                 */
   double sigmaT4;      /*  sigma * T^4                                  */
   double radius;       /*  radius of inner disk implied by its T and L  */
};

struct adcpars{
   double L;            /*  ADC luminosity                               */
   double height;       /*  Height of point-approx ADC above the disk    */
};

struct thirdlightparams {
   double orbphase;     /*  orbital phase at which 3rd light specified   */
   long nbands;         /*  number of bands with 3rd light defined       */
   char filter[MAXBANDPASSES][20];  
                        /* the filter name or SQUARE                     */
   double minlambda[MAXBANDPASSES]; 
                        /* min lambda for a square bandpass              */
   double maxlambda[MAXBANDPASSES]; 
                        /* max lambda for a square bandpass              */
   double fraction[MAXBANDPASSES];     
                        /*  fraction of the total light from 3rd light   */
   double addFlux[MAXBANDPASSES];
                        /* third light flux to added to each bandpass    */
};

struct XYGrid {
   long Nxtiles;        /*  number of tiles in the x direction           */
   long Nztiles;        /*  number of tiles in the y direction           */
   double deltax;       /*  tile spacing in the x direction              */
   double deltaz;       /*  tile spacing in the y direction              */
   double deltal;       /*  sqrt( deltax**2 + deltaz**2)                 */
   double xmin;         /*  edge of the grid in the -x direction         */
   double xmax;         /*  edge of the grid in the  x direction         */
   double ymin;         /*  edge of the grid in the -y direction         */
   double ymax;         /*  edge of the grid in the  y direction         */
   double zmin;         /*  edge of the grid in the -z direction         */
   double zmax;         /*  edge of the grid in the  z direction         */
   double Topy[GRIDXTILES][GRIDZTILES];    /* maximum and minimum values */
   double Bottomy[GRIDXTILES][GRIDZTILES]; /* of y at each grid point    */
};

struct dataparams {
   long nbands;         /* number of light curves                        */
   char filename[MAXBANDPASSES][30];
                        /*  name of file containing the observed fluxes  */
   char filter[MAXBANDPASSES][20];
                        /* the filter name or SQUARE                     */
   double minlambda[MAXBANDPASSES]; 
                        /* min lambda for a square bandpass              */
   double maxlambda[MAXBANDPASSES]; 
                        /* max lambda for a square bandpass              */
   long npoints[MAXBANDPASSES]; 
                        /* number of points in the lightcurve            */
   double phase[MAXBANDPASSES][MAXDATAPOINTS];
                        /* orbital phases of the data points             */
   double flux[MAXBANDPASSES][MAXDATAPOINTS];
                        /* observed fluxes                               */
   double standdev[MAXBANDPASSES][MAXDATAPOINTS];
                        /* standard deviations of the fluxes             */
   double chisquare;    /* chi^2 of the fit                              */
};


/*******************************************************
*
*   Function prototypes
*
*********************************************************/

/*  MAIN.C                             */
void Initialize( char *parfilename );
void CalcSysPars( void );

/*  INPUT.C                            */
void ReadInput( void );
void ReadPars( void );
void CheckPars( void );
void ReadGDTable( void );
void ReadLDTable( void );
void ReadIperpTable( void );
void ReadIBBfilterTable( void );
void ReadZzetaTable( void );

/*  STAR1.C                            */
double Star1TotFlux( double distance );
double Star1Flambda( char *filter, double minlambda, double maxlambda );
double MeanRocheRadius( double q );

/*  STAR2.C                            */
void MakeStar2Tiles( void );
double V( double r, double theta, double phi);
struct SphereVector gradV ( double r, double theta, double phi);
struct SphereVector Normal( struct SphereVector delV);
double FindL1( void );
double findR( double Vtarget, double theta, double phi);
double Star2TopY( double x, double z);
double TileArea( double r, double theta, struct SphereVector normals,
                 double dtheta, double dphi);
struct SphereVector dStodA( struct SphereVector norm, double deltaS);
double GetGDbeta( void );
double ClaretHmu( double T, double logg, char*filter, double mu);
double GetIperp( double T, double logg, char *filter);
double Star2L( void );

/*  DISKGEOM.C                         */
void MakeDiskTiles ( void );
double DiskTopH( double a, double zeta );
double DiskBottomH( double a, double zeta );
double MainDiskH( double a );
double DiskRimH( double a, double zeta );
double DiskTorusH( double a, double zeta );
struct CylVector DiskNormal( char *where, double a1, double a2,
			                  double zeta1, double zeta2);
double TopTileArea( long itile, double a1, double a2, 
		    double zeta1, double zeta2);
double EdgeTileArea( long itile, double zeta1, double zeta2, double dh);
double RhoToA( double rho, double zeta );
double AToRho( double a, double rho );
void MakeDiskRim( void );
void MakeDiskTorus( void );

/*    DISKFLUX.C                       */
double DiskTopT( double a, double zeta );
double MainDiskT( double a, double zeta );
double DiskRimT( double a, double zeta );
double DiskTorusT( double a, double zeta );
double DiskEdgeT( double zeta );
double DiskL( void );
double InnerDiskTotFlux( double distance);
double InnerDiskFlambda( char *filter, double minlambda, double maxlambda);
double ADCTotFlux( double distance);


/*    LIGHTCURVE.C                     */
void MakeLightCurves( void );
void FluxesAtPhase( double phase, double TotalFlux[] );
void ThirdLight( void );
void Normalize( void );
void Irradiate( void );
void MakeYlimits( void );
double Transmission( struct CartVector start, struct CartVector end );
double EscapeFraction( struct CartVector start, 
                                struct CartVector direction);

/*  UTILITY.C                          */
double AngleDistance( double theta1, double phi1,
                      double theta2, double phi2);
double CartDotProd( struct CartVector A, struct CartVector B);
struct SphereVector Cart2Sphere( struct CartVector Acart,
                       double theta, double phi );
struct CartVector Sphere2Cart( struct SphereVector Asphere, 
                       double theta, double phi );
struct CartVector Cyl2Cart( struct CylVector Asphere, double zeta );
double Planck( char * mode, double temperature, double lambdanu );
double BBSquareIntensity( double T, double lowlambda, double highlambda);
double BBFilterIntensity( double T, char *filter);

/*  FITDATA.C                          */
void ReadData( long band);

/*  OUTPUT.C                           */
void Quit( char *outputline);
void WriteResults( void );
void WriteLightCurves( void );
void WriteSysPars( void );

/*  DIAGNOSE.C                         */
void InspectInput( void );
void WritePars( void );
void WriteGDTable( void );
void WriteLDTable( void );
void WriteIperpTable( void );
void WriteIBBfilterTable( void );
void WriteZzetaTable( void );
void InspectStar2Tiles( void );
void InspectDiskTiles( double targetarea, 
                       long nringMain, long maintiles, 
                       long nringEdge, long edgetiles );
void InspectYlimits(void);
void InspectEscape( struct CartVector sunvector, char *filter,
                    double Star1Emitted, double Star1Escape,
                    double T2mu[], double T2Emitted[], double T2Escape[],
                    double TDiskmu[], double TDiskEmitted[],
                    double TDiskEscape[] );
void InspectHeatDiskBy1(double TDiskTold[], double muA1toD[],
                        double DeltaT41toD[], double transmit1toD[] );
void InspectHeatDiskByID(double TDiskTold[], double muAidtoD[],
			 double DeltaT4idtoD[], double transmitidtoD[] );
void InspectHeat2By1( double T2Told[], double muA1to2[], 
		      double DeltaT41to2[], double transmit1to2[] );
void InspectHeat2ByID( double T2Told[], double muAidto2[], 
                       double DeltaT4idto2[], double transmitidto2[] );
void InspectHeat2ByDisk( double T2Told[], double DeltaT4Dto2[], double T2T[] );
void InspectHeatDiskByADC( char *whatside,
                           double TDiskTold[], double muAADCtoD[],
			   double DeltaT4ADCtoD[], double transmitADCtoD[] );
void InspectHeat2ByADC( char *whatside,
                        double T2Told[], double muAADCto2[],
                        double DeltaT4ADCto2[], double transmitADCto2[] );
void WriteData( long band );


/*******************************************************
*
*   Global variables
*
*******************************************************/

/*   Just because you are paranoid that does not mean.... */
char verbose[20];

/*   Flow control and system variables */
struct filenames     filename;
struct flowcontrol   control;
struct systemparams  syspars;

/*   Gravity Darkening variables       */
long maxGDindex;
double GDT[30], fourbeta[30];

/*   Limb Darkening variables          */
char LDfilterName[MAXFILTERS][20];
long maxLDgindex, maxLDTindex, maxLDfilterindex;
double LDlogg[20], LDT[100], LDtable[20][100][MAXFILTERS][5];

/*   Iperp variables                   */
char IperpfilterName[MAXFILTERS][20];
long maxIperpgindex, maxIperpTindex, maxIperpfilterindex;
double Iperplogg[20], IperpT[100], Iperptable[20][100][MAXFILTERS];

/*   IBBfilter variables               */
char IBBfilterName[MAXFILTERS][20];
long maxIBBTindex, maxIBBfilterindex;
double IBBTmin, IBBTmax, IBBdeltaT;
double IBBT[150001], IBBtable[150001][MAXFILTERS];

/*   ZBBzeta variables                 */
long maxBBzetaindex;
double deltaBBzeta, BBzetamax, ZBBzeta[40000];

/*   Star 1 variables                  */
struct star1params   star1;
double T1I[MAXBANDPASSES];

/*   Star 2 variables                  */
struct star2params   star2;
struct star2spotparams star2spot;
struct CartVector    T2normCart[MAX2TILES];
struct SphereVector     T2gradV[MAX2TILES], T2normSphere[MAX2TILES];
double  T2r[MAX2TILES], T2theta[MAX2TILES],        T2phi[MAX2TILES], 
        T2x[MAX2TILES],     T2y[MAX2TILES],          T2z[MAX2TILES],
        T2g[MAX2TILES],  T2logg[MAX2TILES],
       T2dS[MAX2TILES],     T2T[MAX2TILES], 
        T2I[MAXBANDPASSES][MAX2TILES];

/*   Disk variables                    */
struct wholediskpars disk;
struct maindiskpars  maindisk;
struct diskedgepars  diskedge;
struct diskrimpars   diskrim;
struct disktoruspars disktorus;
struct innerdiskpars innerdisk;
struct diskspotpars  diskspot;
struct CylVector            TDisknormCyl[MAXDISKTILES];
struct CartVector          TDisknormCart[MAXDISKTILES];
double TDiskRho[MAXDISKTILES], TDiskZeta[MAXDISKTILES],  TDiskH[MAXDISKTILES], 
         TDiskx[MAXDISKTILES],    TDisky[MAXDISKTILES],  TDiskz[MAXDISKTILES], 
        TDiskdS[MAXDISKTILES],    TDiskT[MAXDISKTILES], TDiskT4[MAXDISKTILES],
        TDiska[MAXDISKTILES],             TDiskI[MAXBANDPASSES][MAXDISKTILES];

/*   ADC variable   */
struct adcpars adc;

/*   Light curve variables             */
struct orbitparams   orbit;
struct XYGrid Grid;
struct thirdlightparams thirdlight;
double LCphase[MAXPHASES], LCflux[MAXBANDPASSES][MAXPHASES], 
                           NormLC[MAXBANDPASSES][MAXPHASES];

/*  Variables concerning observational data */
struct dataparams data;
