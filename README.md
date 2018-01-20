# Concrete

This is the accompanying code to the paper http://www.tandfonline.com/doi/abs/10.1080/15598608.2016.1158675, about variance estimation of a U-statistic for the learning performance on the concrete dataset of the UCI Machine Learning Repository.

The underlying data, the two .dat files, contain the first three principal components of the concrete slump dataset (https://archive.ics.uci.edu/ml/datasets/Concrete+Slump+Test).
The code makes use of super-fast linear regression learning, by implementing immediate inversion of symmetric 3-by-3 matrices in C.


# Purpose
The purpose of the paper is to illuminate the existence of a variance estimator of cross-validation.  In the paper, we explain why such a variance estimator exists if the learning sample size does not exceed half of the total sample size minus one. This repository contains an implementation of such a variance estimator.

The sample size in the dataset is 103.

So, the maximal sample size allowing for a variance estimator is 51.


# Compilation
requires presence of the gsl library. On debian-related systems, gsl is installed using

sudo apt-get install libgsl10-dev

Then, execute the commands

make
sudo make install

and enjoy.

Needs 3 command line arguments: the number of resample in each iteration (as high as possible, try at least 11e4 or 1e5, a random seed, and the learning set size (needs to be between 3 and 51.)

Output: The leave (n-g)-out estimator, the estimator for Theta^2, and the variance estimator for the leave-p-out estimator.



