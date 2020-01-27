require(ConfIntVariance)

d  <- c(2, 2, 4, 6, 2, 6, 4, 5, 4, 6)

## first example: the mean.
U <- mean(d)
                                        # the kernel for this U-statistic
kern  <- function(x) x
mean(apply(combn(10, 2), 2, function(y) kern(d[y])))

                                        # best estimator for theta squared
                                        # = 16.53333
ts  <- mean(apply(combn(10, 2), 2, function(y) kern(d[y[1]]) * kern(d[y[2]])))

                                        # estimator for variance of the mean
varmean  <- U * U - ts

                                        # this is the same: the estimated variance of the U-statistic
var(d) / 10

                                        # correspondingly, the confidence interval for the true mean:
                                        # the t-quantile at 1-0.05/2
q  <- qt(df=length(d) -1 , p=1- 0.05/2)

confIntLower  <-  U - q * sqrt(varmean)tt
confIntUpper  <-  U + q * sqrt(varmean)

                                        # this confidence interval coincides with the classical one
print(t.test(d))


## second example: the variance
U  <- var(d)
                                        # using the way to compute it as a U-statistic
mean(apply(combn(10, 2), 2, function(y) kern(d[y])))

kern  <- function(x) (x[1]-x[2])^2 / 2
                                        # just to check correctness: best estimator for theta squared:
ts  <- mean(apply(combn(10, 4), 2, function(y)
(
    kern(d[y[c(1, 2)]]) * kern(d[y[c(3, 4)]])
    + kern(d[y[c(1, 3)]]) * kern(d[y[c(2, 4)]])
    + kern(d[y[c(1, 4)]]) * kern(d[y[c(2, 3)]])
    ) / 3
)
)
                                        # estimator for variance of the U-statistic
varvar  <- U * U - ts

                                        # correspondingly, the confidence interval for the true mean:
                                        # the t-quantile at 1-0.05/2
q  <- qt(df=length(d) -1 , p=1- 0.05/2)

confIntLower  <-  U - q * sqrt(varvar)
confIntUpper  <-  U + q * sqrt(varvar)

                                        # this confidence interval coincides with the R-package
print(varwci(d))

