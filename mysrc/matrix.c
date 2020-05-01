#include "image.h"
#include "matrix.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

matrix make_identity_homography()
{
    matrix H = make_matrix(3,3);   
    H.data[0] = 1;
    H.data[4] = 1;
    H.data[8] = 1;
    return H;
}

matrix make_translation_homography(float dx, float dy)
{
    matrix H = make_identity_homography();
    H.data[2] = dx;
    H.data[5] = dy;
    return H;
}
matrix make_identity(int rows, int cols)
{
    int i,j=MIN(rows,cols);
    matrix m = make_matrix(rows, cols);
	
	
    for(i = 0; i < j; ++i){
        m.data[i*cols+i] = 1;
	}
    return m;
}

// Make empty matrix filled with zeros
// int rows: number of rows in matrix
// int cols: number of columns in matrix
// returns: matrix of specified size, filled with zeros
matrix make_matrix(int rows, int cols)
{
    matrix m;
    m.rows = rows;
    m.cols = cols;
    m.shallow = 0;
    m.data = calloc(m.rows*m.cols, sizeof(float));
    return m;
}

// Make a matrix with uniformly random elements
// int rows, cols: size of matrix
// float s: range of randomness, [-s, s]
// returns: matrix of rows x cols with elements in range [-s,s]
matrix random_matrix(int rows, int cols, float s)
{
    matrix m = make_matrix(rows, cols);
    int i, j,im;
    for(i = 0; i < rows; ++i){
		im=i*cols;
        for(j = 0; j < cols; ++j)
			m.data[im + j] = 2*s*((float)rand()/RAND_MAX) - s;    
        
    }
    return m;
}

// Free memory associated with matrix
// matrix m: matrix to be freed
void free_matrix(matrix *m)
{
    if (!m->shallow && m->data){
		free(m->data);m->data=NULL;
	}
}
// Copy a matrix
// matrix m: matrix to be copied
// returns: matrix that is a deep copy of m
matrix copy_matrix(matrix m)
{
    matrix c = make_matrix(m.rows, m.cols);
    // TODO: 1.1 - Fill in the new matrix
	memcpy(c.data,m.data,sizeof(float)*m.rows*m.cols);
    return c;
}

// Transpose a matrix
// matrix m: matrix to be transposed
// returns: matrix, result of transposition
matrix transpose_matrix(matrix m)
{
    // TODO: 1.2 - Make a matrix the correct size, fill it in
    matrix t = make_matrix(m.cols,m.rows);
    int i, j,it;
    for(i = 0; i < t.rows; ++i){
		it=i*t.cols;
        for(j = 0; j < t.cols; ++j)t.data[it+j] = m.data[j*m.cols+i];
    }
    return t;
}

// Perform y = ax + y
// float a: scalar for matrix x
// matrix x: left operand to the scaled addition
// matrix y: unscaled right operand, also stores result
void axpy_matrix(float a, matrix x, matrix y)
{
    assert(x.cols == y.cols);
    assert(x.rows == y.rows);
    // TODO: 1.3 - Perform the weighted sum, store result back in y
	int i;
	#pragma omp parallel for
	for(i=0;i<x.rows*x.cols;++i)y.data[i]+=a*x.data[i];
}

// Perform matrix multiplication a*b, return result
// matrix a,b: operands
// returns: new matrix that is the result
matrix chainTrace(int i,int j,image s,matrix *p){
	if(i==j)return p[i];
	matrix a=chainTrace(i,get_pixel(s,i,j,1),s,p);
	matrix b=chainTrace(get_pixel(s,i,j,1)+1,j,s,p);
	matrix c=matmul(a,b);
	FM(a);FM(b);
	return c;
}

matrix matchain(matrix * chain,int n){
	int i,r,j,k;
	for(i=1;i<n;++i)assert(chain[i].rows==chain[i-1].cols);
	image s=make_image(n,n,2);
	for(r=2;r<n+1;++r){
		for(i=0;i<n-r+1;++i){
			j=i+r-1;
			set_pixel(s,i,j,0,get_pixel(s,i+1,j,0)+chain[i].rows*chain[i].cols*chain[j].cols);
			set_pixel(s,i,j,1,i);
			for(k=i+1;k<j;++k){
				int t=get_pixel(s,i,k,0)+get_pixel(s,k+1,j,0)+chain[i].rows*chain[k].cols*chain[j].cols;
				if(t<get_pixel(s,i,j,0)){set_pixel(s,i,j,0,t);set_pixel(s,i,j,1,k);}
			}
		}
	}
	matrix m=chainTrace(0,n-1,s,chain);
	FIMG(s);
	return m;
}

matrix matmax(int a, matrix b)
{
	assert(b.rows%a==0);
    int i, j, k,ip;
	float tmp;
    matrix p = make_matrix(b.rows/a, b.cols);
	for(i = 0; i < p.rows; ++i){
		ip=i*p.cols;
		for(j = 0; j < p.cols; ++j){
			tmp=b.data[i*a*b.cols+j];
			for(k = 1; k < a; ++k){
				if(b.data[(i*a+k)*b.cols+j]>tmp)tmp=b.data[(i*a+k)*b.cols+j];
			}
			p.data[ip+j]=tmp;
		}
	}
    return p;
}

matrix matmul_single(matrix a, matrix b)
{
    // TODO: 1.4 - Implement matrix multiplication. Make sure it's fast!
	assert(a.cols == b.rows);
    int i, j, k,ip,ia;
	double tmp;
    matrix p = make_matrix(a.rows, b.cols);
	for(i = 0; i < p.rows; ++i){
		ip=i*p.cols;ia=i*a.cols;
		for(j = 0; j < p.cols; ++j){
			tmp=0;
			for(k = 0; k < a.cols; ++k)tmp+= a.data[ia+k]*b.data[k*b.cols+j];
			p.data[ip+j]=tmp;
		}
	}
    return p;
}

matrix matmul(matrix a, matrix b)
{
    // TODO: 1.4 - Implement matrix multiplication. Make sure it's fast!
	assert(a.cols == b.rows);
    int i, j, k,ip,ia;
	double tmp;
    matrix p = make_matrix(a.rows, b.cols);
	#pragma omp parallel for
	for(i = 0; i < p.rows; ++i){
		ip=i*p.cols;ia=i*a.cols;
		for(j = 0; j < p.cols; ++j){
			tmp=0;
			for(k = 0; k < a.cols; ++k)tmp+= a.data[ia+k]*b.data[k*b.cols+j];
			p.data[ip+j]=tmp;
		}
	}
    return p;
}
matrix mathamm(matrix a, matrix b,OPT o)
{
    assert(a.cols == b.cols);
    assert(a.rows == b.rows);
    int i;
    matrix p = make_matrix(a.rows, a.cols);
	switch(o){
		case MUL:
			
			for(i = 0; i < p.rows*p.cols; ++i)p.data[i] = a.data[i] * b.data[i];
			break;
		case SUB:
			
			for(i = 0; i < p.rows*p.cols; ++i)p.data[i] = a.data[i] - b.data[i];
			break;
		case PLUS:
			
			for(i = 0; i < p.rows*p.cols; ++i)p.data[i] = a.data[i] + b.data[i];
			break;
	}
    return p;
}
// In-place, element-wise scaling of matrix
// float s: scaling factor
// matrix m: matrix to be scaled
void scal_matrix(float s, matrix m,OPT o)
{
    int i;
	switch(o){
		case MUL:
			
			for(i = 0; i < m.rows*m.cols; ++i)m.data[i]*=s;
			break;
		case SUB:
			
			for(i = 0; i < m.rows*m.cols; ++i)m.data[i]-=s;
			break;
		case PLUS:
			
			for(i = 0; i < m.rows*m.cols; ++i)m.data[i]+=s;
			break;
	}
}

// Print a matrix
void print_matrix(matrix m)
{
    int i, j;
    printf(" __");
    for(j = 0; j < 16*m.cols-1; ++j) printf(" ");
    printf("__ \n");

    printf("|  ");
    for(j = 0; j < 16*m.cols-1; ++j) printf(" ");
    printf("  |\n");

    for(i = 0; i < m.rows; ++i){
        printf("|  ");
        for(j = 0; j < m.cols; ++j){
            printf("%15.7f ", m.data[i*m.cols + j]);
        }
        printf(" |\n");
    }
    printf("|__");
    for(j = 0; j < 16*m.cols-1; ++j) printf(" ");
    printf("__|\n");
}

// Used for matrix inversion
double** augment_matrix(matrix m)
{
    int i,j,im;
    //~ matrix c = make_matrix(m.rows, m.cols*2);
	double **c=calloc(m.rows,sizeof(double *));
	for(i=0;i<m.rows;++i)c[i]=calloc(2*m.cols,sizeof(double));
	
    for(i = 0; i < m.rows; ++i){
		im=i*m.cols;
        for(j = 0; j < m.cols; ++j){
            c[i][j] = m.data[im + j];
        }
    }
	
    for(j = 0; j < m.rows; ++j){
        c[j][j+m.cols] = 1;
    }
    return c;
}

// Invert matrix m
matrix matrix_invert(matrix m)
{
    int i, j, k;
    //print_matrix(m);
    matrix none = {0};
    if(m.rows != m.cols){
        fprintf(stderr, "Matrix not square\n");
        return none;
    }
    double **cdata = augment_matrix(m);
    //print_matrix(c);
    //~ float **cdata = calloc(c.rows, sizeof(float *));
    //~ for(i = 0; i < c.rows; ++i){
        //~ cdata[i] = c.data + i*c.cols;
    //~ }
		
    for(k = 0; k < m.rows; ++k){
        double p = 0.;
        int index = -1;
        for(i = k; i < m.rows; ++i){
            double val = fabs(cdata[i][k]);
            if(val > p){
                p = val;
                index = i;
            }
        }
        if(index == -1){
            //fprintf(stderr, "Can't do it, sorry!\n");
            // FM(c);
			for(i=0;i<m.rows;++i)free(cdata[i]);
			free(cdata);
            return none;
        }

        double *swap = cdata[index];
        cdata[index] = cdata[k];
        cdata[k] = swap;

        double val = cdata[k][k];
        cdata[k][k] = 1;
        for(j = k+1; j < 2*m.cols; ++j){
            cdata[k][j] /= val;
        }
        for(i = k+1; i < m.rows; ++i){
            double s = -cdata[i][k];
            cdata[i][k] = 0;
            for(j = k+1; j < 2*m.cols; ++j){
                cdata[i][j] +=  s*cdata[k][j];
            }
        }
    }
    for(k = m.rows-1; k > 0; --k){
        for(i = 0; i < k; ++i){
            double s = -cdata[i][k];
            cdata[i][k] = 0;			
            for(j = k+1; j < 2*m.cols; ++j){
                cdata[i][j] += s*cdata[k][j];
            }
        }
    }
    //print_matrix(c);
    matrix inv = make_matrix(m.rows, m.cols);
	int im;
	
    for(i = 0; i < m.rows; ++i){
		im=i*m.cols;
        for(j = 0; j < m.cols; ++j){
            inv.data[im + j] = cdata[i][j+m.cols];
        }
    }
    //~ FM(c);
	for(i=0;i<m.rows;++i)free(cdata[i]);
    free(cdata);
    //print_matrix(inv);
    return inv;
}

matrix solve_system(matrix M, matrix b)
{
    matrix none = {0};
    matrix Mt = transpose_matrix(M);
    matrix MtM = matmul(Mt, M);
    matrix MtMinv = matrix_invert(MtM);
	FM(MtM);
    if(!MtMinv.data) {FM(Mt);return none;}
    //~ matrix Mdag = matmul(MtMinv, Mt);
	//~ FM(Mt);FM(MtMinv);
    //~ matrix a = matmul(Mdag,b);
	//~ FM(Mdag);
	b.shallow=1;
	matrix c[]={MtMinv,Mt,b};
    return matchain(c,3);
}

float *matmulv(matrix m, float *v)
{
    float *p = calloc(m.rows, sizeof(float));
    int i, j,im;
	double tmp;
	
    for(i = 0; i < m.rows; ++i){
		im=i*m.cols;
		tmp=p[i];
        for(j = 0; j < m.cols; ++j) tmp += m.data[im+j]*v[j];
        p[i]=tmp;
    }
    return p;
}
float *LUP_solve(matrix L, matrix U, int *p, float *b)
{
    int i, j,iL,iU;
    float *c = calloc(L.rows, sizeof(float));
	double tmp;
	
    for(i = 0; i < L.rows; ++i){
        int pi = p[i];
        c[i] = b[pi];
		iL=i*L.cols;
		tmp=c[i];
        for(j = 0; j < i; ++ j)tmp-= L.data[iL+j]*c[j];
		c[i]=tmp;
    }
	
    for(i = U.rows-1; i >= 0; --i){
		iU=i*U.cols;
		tmp=c[i];
        for(j = i+1; j < U.cols; ++j){
            tmp -= U.data[iU+j]*c[j];
        }
        c[i] =tmp/U.data[iU+i];
    }
    return c;
}
int* in_place_LUP(matrix m)
{
    int *pivot = calloc(m.rows, sizeof(int));
    if(m.rows != m.cols){
        fprintf(stderr, "Matrix not square\n");
        return 0;
    }	
    int i, j, k,im;

	float **mdata = calloc(m.rows, sizeof(float *));
    for(i = 0; i < m.rows; ++i) mdata[i] = m.data + i*m.cols;
    for(k = 0; k < m.rows; ++k) pivot[k] = k;
	
    for(k = 0; k < m.rows; ++k){
        double p = 0.;
        int index = -1;
        for(i = k; i < m.rows; ++i){
            double val = fabs(mdata[i][k]);
            if(val > p){
                p = val;
                index = i;
            }
        }
        if(index == -1){
            fprintf(stderr, "Matrix is singular\n");
            return 0;
        }

        int swapi = pivot[k];
        pivot[k] = pivot[index];
        pivot[index] = swapi;

        float *swap = mdata[index];
        mdata[index] = mdata[k];
        mdata[k] = swap;
        for(i = k+1; i < m.rows; ++i){
            mdata[i][k] /= mdata[k][k];
            for(j = k+1; j < m.cols; ++j){
                mdata[i][j] -= mdata[i][k] * mdata[k][j];
            }
        }
    }
	
	float *data=calloc(m.rows*m.cols, sizeof(float));
	
	for(i=0;i<m.rows;++i){
		im=i*m.cols;
		for(j=0;j<m.cols;++j)data[im+j]=mdata[i][j];
	}
	free(mdata);
	FM(m);
	m.data=data;
    return pivot;
}
double mag_matrix(matrix m)
{
    int i;
    double sum = 0;
    for(i = 0; i < m.rows*m.cols; ++i)
	sum +=pow(m.data[i],2);
    return sqrt(sum);
}

float *sle_solve(matrix A, float *b)
{
    int *p = in_place_LUP(A);
    return LUP_solve(A, A, p, b);
}

void test_matrix()
{
    int i;
    for(i = 0; i < 100; ++i){
        int s = rand()%4 + 3;
        matrix m = random_matrix(s, s, 10);
        matrix inv = matrix_invert(m);
        if(inv.data){
            matrix res = matmul(m, inv);
            print_matrix(res);
        }
    }
}