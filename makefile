
OBJS = main.o input.o star1.o star2.o diskgeom.o diskflux.o lightcurve.o output.o utility.o fitdata.o diagnose.o

XRbinary :  $(OBJS)
	cc -lm -o a.out $(OBJS)

main.o :  main.c header.h
	cc -c main.c

input.o :  input.c header.h
	cc -c input.c

star1.o :  star1.c header.h
	cc -c star1.c

star2.o :  star2.c header.h
	cc -c star2.c

diskgeom.o :  diskgeom.c header.h
	cc -c diskgeom.c

diskflux.o :  diskflux.c header.h
	cc -c diskflux.c

lightcurve.o :  lightcurve.c header.h
	cc -c lightcurve.c

output.o :  output.c header.h
	cc -c output.c

utility.o :  utility.c header.h
	cc -c utility.c

fitdata.o :  fitdata.c header.h
	cc -c fitdata.c

diagnose.o : diagnose.c header.h
	cc -c diagnose.c
