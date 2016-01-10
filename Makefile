all:
	gcc -std=c99 -Wall -Ofast -I/usr/local/include -c concrete.c
	gcc -Ofast -L/usr/local/lib concrete.o -lgsl -lgslcblas -lm -o concrete


