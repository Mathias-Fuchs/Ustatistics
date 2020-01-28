#include "runningStats.h"


rs rs_init() {
	rs r;
	r.m_n = 0; // to be safe
	return r;
}

void rs_clear(rs* r) {
	r->m_n = 0;
}

void rs_push(rs* r, double x) {
	r->m_n++;

	// See Knuth TAOCP vol 2, 3rd edition, page 232
	if (r->m_n == 1)
	{
		r->m_oldM = r->m_newM = x;
		r->m_oldS = 0.0;
	}
	else
	{
		r->m_newM = r->m_oldM + (x - r->m_oldM) / r->m_n;
		r->m_newS = r->m_oldS + (x - r->m_oldM) * (x - r->m_newM);

		// set up for next iteration
		r->m_oldM = r->m_newM;
		r->m_oldS = r->m_newS;
	}
}


double rs_mean(const rs* r)
{
	return (r->m_n > 0) ? r->m_newM : 0.0;
}

double rs_variance(const rs* r)
{
	return ((r->m_n > 1) ? r->m_newS / (r->m_n - 1) : 0.0);
}

