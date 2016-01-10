# concrete
Variance estimation of a U-statistic for the learning performance on the concrete dataset of the UCI Machine Learning Repository


The underlying data, the two .dat files, contain the first three principal components of the concrete slump dataset (https://archive.ics.uci.edu/ml/datasets/Concrete+Slump+Test).
The code makes use of super-fast linear regression learning, by implementing immediate inversion of symmetric 3-by-3 matrices in C.


# compilation
requires presence of the gsl library. On debian-related systems, gsl is installed using

sudo apt-get install libgsl10-dev

Then, execute the commands

make
sudo make install

and enjoy.
