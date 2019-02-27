
# Overview
The notion of U-statistic was introduced in a seminal paper by Wassiliy Hoeffding in 1948 and has matured to constitute one of the building blocks of modern statistics.

The importance of the concept lies in the fact that many interesting statistics turn out to be part of the class of U-statistics in disguise. In particular, the sample mean and variance, but also the (complete) cross-validation estimator of the error in supervised machine learning.

This library contains a handful of functions for computation of U-statistics, and in particular contains code for the computation confidence interval.
More precisely, one can only approximate a U-statistic due to the high number of terms in its definition. Therefore, it is desirable to know if the approximation is reliable, and that can be done with a confidence interval.

Its meaning is that in at least 95% of all cases the computed confidence interval will contain the true value of the U-statistic.

This confidence interval is not to be confused with the confidence interval for &theta;, the estimation target of the U-statistic itself.
For instance, if the U-statistic is the sample mean, then the &theta; is the population mean, and the confidence interval is the usual confidence interval for the mean, as in, for instance, the "t.test" function in R.)

This library is capable of computing both confidence intervals.

The core function is the function U defined in the header file U.h.
It expects the kernel of the U-statistic as a callback function.

The library builds under Visual Studio in Windows, and under Debian/Ubuntu. The only requirement is the gnu scientific library. Under Windows, it can easily be installed using the vcpkg library.

An important example is the concrete dataset which was used as the data example in the paper https://epub.ub.uni-muenchen.de/27656/7/TR.pdf (published as https://www.tandfonline.com/doi/abs/10.1080/15598608.2016.1158675)


Pull request with more examples for U-statistics are welcome!

# Importance
One of the most important applications is to supervised learning cross-validation. In fact, the following papers explain in which sense the cross-validated error rate is a U-statistic.
Therefore, a confidence interval for the machine learning error rate can be obtained by a confidence interval for the 

- https://epub.ub.uni-muenchen.de/17654/
- https://epub.ub.uni-muenchen.de/27656/7/TR.pdf published as https://www.tandfonline.com/doi/abs/10.1080/15598608.2016.1158675
- http://www3.stat.sinica.edu.tw/sstest/j24n3/j24n34/j24n34.html

# Concrete

An example dataset is the concrete slump dataset http://archive.ics.uci.edu/ml/datasets/concrete+slump+test

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
Estimating *theta* is done with a U-statistic whose kernel is implemented in the function kernelTheta.
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


