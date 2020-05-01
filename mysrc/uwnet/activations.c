#include <assert.h>
#include <math.h>
#include "uwnet.h"
// Run an activation function on each element in a matrix,
// modifies the matrix in place
// matrix m: Input to activation function
// ACTIVATION a: function to run
void activate_matrix(matrix m, ACTIVATION a)
{
    int i, j,id,idx;
    for(i = 0; i < m.rows; ++i){
        double sum = 0;
		id=i*m.cols;
        for(j = 0; j < m.cols; ++j){
			idx= id+ j;
            switch(a){
				case LOGISTIC: 	m.data[idx]=1/(1+exp(-m.data[idx]));break;
				case RELU:     	if(m.data[idx]<0) m.data[idx]=0;	break;
				case LRELU:		if(m.data[idx]<0) m.data[idx]*=0.1;	break;
				case SOFTMAX: 	m.data[idx]=exp(m.data[idx]);sum +=m.data[idx];break;
				case LINEAR:break;
			}
        }
        if (a == SOFTMAX) {
            // TODO: have to normalize by sum if we are using SOFTMAX
			if(sum){
				sum=1/sum;
				for(j = 0; j < m.cols; ++j)m.data[id+j]*=sum;
			}
        }
    }
}

// Calculates the gradient of an activation function and multiplies it into
// the delta for a layer
// matrix m: an activated layer output
// ACTIVATION a: activation function for a layer
// matrix d: delta before activation gradient
void gradient_matrix(matrix m, ACTIVATION a, matrix d)
{
    int i, j;
	double x,g;
    for(i = 0; i < m.rows; ++i){
        for(j = 0; j < m.cols; ++j){
            x = m.data[i*m.cols + j];
            // TODO: multiply the correct element of d by the gradient
			g=0;
			switch(a){
				case LOGISTIC:	g=x*(1-x);		break;
				case RELU:		g=(x>0)?1:0;	break;
				case LRELU:		g=(x>0)?1:0.1;	break;
				case LINEAR:
				case SOFTMAX:	g=1;			break;
			}
			d.data[i*d.cols+j]*=g;
        }
    }
}
