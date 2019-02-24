# Concrete

This is the accompanying code to the paper http://www.tandfonline.com/doi/abs/10.1080/15598608.2016.1158675, about variance estimation of a U-statistic for the learning performance on the concrete dataset of the UCI Machine Learning Repository.

The underlying data, the two .dat files, contain the first three principal components of the concrete slump dataset (https://archive.ics.uci.edu/ml/datasets/Concrete+Slump+Test).
The code makes use of super-fast linear regression learning, by implementing immediate inversion of symmetric 3-by-3 matrices in C.

# Purpose
The purpose of the paper is to illuminate the existence of a variance estimator of cross-validation.
In the paper, we explain why such a variance estimator exists if the learning sample size does not exceed half of the total sample size minus one. This repository contains an implementation of such a variance estimator.
The sample size in the dataset is 103.
So, the maximal sample size allowing for a variance estimator is 51.

# Background
I am trying to explain what all this is about in a series of online diary entries at  http://www.mathiasfuchs.de/b2.html,  and subsequent entries.

This repository contains code that re-samples a kernel of a U-statistic often enough to obtain a good approximation of the atactual value of the U-statistic.

In particular, this is applied to the problem of estimating the mean square loss of linear regression where both learning and testing are random.

We denote by *theta* the expectation of the mean square of linear regression.
Estimating *theta* is done with a U-statistic whose kernel is implemented in the function gamma.
Let us abbreviate the leave-p-out estimator of theta with TH (for theta-hat.)

Likewise, the  estimator of its variance is given by the difference of two different U-statistics:

- the one that estimates the expectation of the square of TH.
- the one that estimates the square of the expectation of TH, i.e. the square of theta.

The first of those two is easy: it is already optimally estimated by the square of TH.
The main purpose of this repository is to provide code that estimates the second optimally with a U-statistic.

Its kernel is implemented in the function kernelforthetasquared.

The entire program then computes the estimated variance of the mean square loss of linear regression.
 


# Compilation
In visual studio, just open the folder and use vcpkg to install the single dependency gsl.
There are two executable targets defined in the cmake configuration file, one for the concrete dataset, and one for linear regression on random data.

On debian-related systems, gsl is installed using

sudo apt install libgsl-dev

Then, execute the commands

make
sudo make install

and enjoy.


