#include "uwnet.h"

#include <stdlib.h>
#include <assert.h>
#include <math.h>

matrix mean(matrix x, int spatial)
{
    matrix m = make_matrix(1, x.cols/spatial);
    int i, j;
    for(i = 0; i < x.rows; ++i){
        for(j = 0; j < x.cols; ++j){
            m.data[j/spatial] += x.data[i*x.cols + j];
        }
    }
    for(i = 0; i < m.cols; ++i){
        m.data[i] = m.data[i] / x.rows / spatial;
    }
    return m;
}

matrix variance(matrix x, matrix m, int spatial)
{
    matrix v = make_matrix(1, x.cols/spatial);
    // TODO: 7.1 - calculate variance
	int i, j;
    for(i = 0; i < x.rows; ++i){
        for(j = 0; j < x.cols; ++j){
            v.data[j/spatial] += pow(x.data[i*x.cols + j]-m.data[j/spatial],2);
        }
    }
    for(i = 0; i < v.cols; ++i){
        v.data[i] = v.data[i] / x.rows / spatial;
    }
    return v;
}

matrix normalize(matrix x, matrix m, matrix v, int spatial)
{
    matrix norm = make_matrix(x.rows, x.cols);
    // TODO: 7.2 - normalize array, norm = (x - mean) / sqrt(variance + eps)
	int i,j;
	float m_v,var;
	for(j = 0; j < x.cols; ++j){
		var=sqrt(v.data[j/spatial]);
		if(var==0)var=1e-6;
		m_v=m.data[j/spatial]/var;		
		for(i = 0; i < x.rows; ++i)
            norm.data[i*x.cols+j]=x.data[i*x.cols+j]/var-m_v;
    }	
    return norm;
    
}

matrix batch_normalize_forward(layer l, matrix x)
{
    float s = .1;
    int spatial = x.cols / l.rolling_mean.cols;
	assert(spatial>0);
    if (x.rows == 1){
        return normalize(x, l.rolling_mean, l.rolling_variance, spatial);
    }
    matrix m = mean(x, spatial);
    matrix v = variance(x, m, spatial);

    matrix x_norm = normalize(x, m, v, spatial);

    scal_matrix(1-s, l.rolling_mean,MUL);
    axpy_matrix(s, m, l.rolling_mean);

    scal_matrix(1-s, l.rolling_variance,MUL);
    axpy_matrix(s, v, l.rolling_variance);

    FM(m);
    FM(v);

    FM(l.x[0]);
    l.x[0] = x;

    return x_norm;
}


matrix delta_mean(matrix d, matrix variance, int spatial)
{
    matrix dm = make_matrix(1, variance.cols);
    // TODO: 7.3 - calculate dL/dmean
	int i, j;
	float var;
    for(j = 0; j < d.cols; ++j){
		var=sqrt(variance.data[j/spatial]);
		if(var==0)var=1e-6;
		for(i = 0; i < d.rows; ++i)
			dm.data[j/spatial] -=d.data[i*d.cols+j]/var;
    }
    return dm;
}

matrix delta_variance(matrix d, matrix x, matrix mean, matrix variance, int spatial)
{
    matrix dv = make_matrix(1, variance.cols);
    // TODO: 7.4 - calculate dL/dvariance
	int i, j;
	float var;
    for(j = 0; j < d.cols; ++j){
		var=2*sqrt(pow(variance.data[j/spatial],3));
		if(var==0)var=1e-6;
		for(i = 0; i < d.rows; ++i)
			dv.data[j/spatial] -=d.data[i*d.cols+j]*(x.data[i*x.cols+j]-mean.data[j/spatial])/var;
    }
    return dv;
}

matrix delta_batch_norm(matrix d, matrix dm, matrix dv, matrix mean, matrix variance, matrix x, int spatial)
{
    int i, j,idx;
    matrix dx = make_matrix(d.rows, d.cols);
    // TODO: 7.5 - calculate dL/dx
	float var;
	for(j = 0; j < d.cols; ++j){
		idx=j/spatial;
		var=sqrt(variance.data[idx]);
		if(var==0)var=1e-6;
		for(i = 0; i < d.rows; ++i)
		dx.data[i*dx.cols+j] =d.data[i*d.cols+j]/var+(dv.data[idx]*2*(x.data[i*x.cols+j]-mean.data[idx])+dm.data[idx])/(spatial*d.rows);
	}
    return dx;
}

matrix batch_normalize_backward(layer l, matrix d)
{
    int spatial = d.cols / l.rolling_mean.cols;
	assert(spatial!=0);
    matrix x = l.x[0];

    matrix m = mean(x, spatial);
    matrix v = variance(x, m, spatial);

    matrix dm = delta_mean(d, v, spatial);
    matrix dv = delta_variance(d, x, m, v, spatial);
    matrix dx = delta_batch_norm(d, dm, dv, m, v, x, spatial);

    FM(m);
    FM(v);
    FM(dm);
    FM(dv);

    return dx;
}
