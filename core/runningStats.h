#pragma once

typedef struct {
	int m_n;
	double m_oldM;
	double m_newM;
	double m_oldS;
	double m_newS;
} rs;

rs rs_init();

void rs_clear(rs* r);

void rs_push(rs* r, double x);

double rs_mean(const rs* r);

double rs_variance(const rs* r);
