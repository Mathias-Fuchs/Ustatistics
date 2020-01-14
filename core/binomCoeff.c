#include "binomCoeff.h"
#include <limits.h>

rr binomialCoefficient(size_t n, size_t k) {
	rr b;
	if (k == 0) {
		b.i = 1; b.isInfty = 0; return b;
	}
	if (k == 1) {
		b.i = n; b.isInfty = 0; return b;
	}
	if (k > n / 2) return binomialCoefficient(n, n - k);
	rr o = binomialCoefficient(n - 1, k - 1);
	if (o.isInfty || o.i > INT_MAX / n * k) {
		b.isInfty = 1; return b;
	}
	b.i = o.i * n / k;
	b.isInfty = 0;
	return b;
}
