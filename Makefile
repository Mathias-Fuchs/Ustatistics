
CFLAGS += -Wall -Wextra -I/usr/include/ -Icore/ `pkg-config --cflags gsl`
LDFLAGS += `pkg-config --static --libs gsl`





coreB:
	gcc -c $(CFLAGS) core/*.c

coreSharedObject:
	rm main.o
	gcc -fPIC -c $(CFLAGS) core/*.c
	gcc -shared *.o -o libustatistics.so
#	cp libmfsl.so /opt/lib -v

standardExamples: coreB
	gcc -c $(CFLAGS) standardExamplesUstatistics/mean.c -o mean.o
	gcc *.o $(LDFLAGS) -o mean
	rm *.o

	gcc -c $(CFLAGS) standardExamplesUstatistics/variance.c -o variance.o
	gcc *.o $(LDFLAGS) -o mean
	rm *.o

randomGaussianB:
	gcc -c $(CFLAGS) randomGaussianExample/main.c -o randomGaussian.o

concreteDatasetExampleB:
	gcc -c $(CFLAGS) concreteDatasetExample/main.c -o concreteDatasetExample.o

randomGaussian: coreB randomGaussianB
	gcc *.o $(LDFLAGS) -o randomGaussianE

concreteDatasetExample: coreB concreteDatasetExampleB
	gcc *.o $(LDFLAGS) -o concreteDatasetExampleE
	cp concreteDatasetExample/*.dat . -iv

concreteDatasetWithSharedObject:
	gcc -c $(CFLAGS) concreteDatasetExample/*.c
	gcc main.o -Wl,-rpath,. -L. -lmfsl $(LDFLAGS) -o sampleAppWithSharedObject
	cp sample_app/resources/bglWires.sqlite . -v	

clean:
	rm *.o
	rm randomGaussianE
	rm concreteDatasetExampleE

