#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "image.h"
#include "matrix.h"
#define SQRT2PI 2.506628
// Frees an array of descriptors.
// descriptor *d: the array.
// int n: number of elements in array.
void free_descriptors(descriptor *d, int n)
{
    int i;
    for(i = 0; i < n; ++i)free(d[i].data);
    free(d);
}

// Create a feature descriptor for an index in an image.
// image im: source image.
// int i: index in image for the pixel we want to describe.
// returns: descriptor for that index.
descriptor describe_index(image im, int i)
{
    int w = 5;
    descriptor d;
    d.p.x = i%im.w;
    d.p.y = i/im.w;
    d.data = calloc(w*w*im.c, sizeof(float));
    d.n = w*w*im.c;
    int c, dx, dy;
    int count = 0;
    // If you want you can experiment with other descriptors
    // This subtracts the central value from neighbors
    // to compensate some for exposure/lighting changes.
    for(c = 0; c < im.c; ++c){
        float cval = im.data[c*im.w*im.h + i];
        for(dx = -w/2; dx < (w+1)/2; ++dx){
            for(dy = -w/2; dy < (w+1)/2; ++dy){
                float val = get_pixel(im, i%im.w+dx, i/im.w+dy, c);
                d.data[count++] = cval - val;
            }
        }
    }
    return d;
}

// Marks the spot of a point in an image.
// image im: image to mark.
// ponit p: spot to mark in the image.
void mark_spot(image im, point p)
{
    int x = p.x;
    int y = p.y;
    int i;
    for(i = -9; i < 10; ++i){
        set_pixel(im, x+i, y, 0, 1);
        set_pixel(im, x, y+i, 0, 1);
        set_pixel(im, x+i, y, 1, 0);
        set_pixel(im, x, y+i, 1, 0);
        set_pixel(im, x+i, y, 2, 1);
        set_pixel(im, x, y+i, 2, 1);
    }
}

// Marks corners denoted by an array of descriptors.
// image im: image to mark.
// descriptor *d: corners in the image.
// int n: number of descriptors to mark.
void mark_corners(image im, descriptor *d, int n)
{
    int i;
    for(i = 0; i < n; ++i){
        mark_spot(im, d[i].p);
    }
}

// Creates a 1d Gaussian filter.
// float sigma: standard deviation of Gaussian.
// returns: single row image of the filter.
image make_1d_gaussian(float sigma)
{
    int hw=6*sigma+1;
	if((hw&1)==0)hw+=1;
	int hf=hw>>1;
	image g=make_image(hw,1,1);
	int i;
	float sig22=1/(2*sigma*sigma);
	float sigpi=1/(SQRT2PI*sigma);
	for(i=0;i<hw;++i)g.data[i]=exp(-(i-hf)*(i-hf)*sig22)*sigpi;
	l1_normalize(g);
    return g;
}

// Smooths an image using separable Gaussian filter.
// image im: image to smooth.
// float sigma: std dev. for Gaussian.
// returns: smoothed image.
image smooth_image(image im, float sigma)
{
    image s,g,tmp;
    if(0){
        g = make_filter(GAUSS,sigma);
        s = convolve_image(im, g, 1);
    } else {
        // TODO: optional, use two convolutions with 1d gaussian filter.
        // If you implement, disable the above if check.
		g=make_1d_gaussian(sigma);
		tmp=convolve_image(im, g, 1);
		g.h=g.w;g.w=1;
		s=convolve_image(tmp,g, 1);
		FIMG(tmp);        
    }
	FIMG(g);
	return s;
}

// Calculate the structure matrix of an image.
// image im: the input image.
// float sigma: std dev. to use for weighted sum.
// returns: structure matrix. 1st channel is Ix^2, 2nd channel is Iy^2,
//          third channel is IxIy.
image structure_matrix(image im, float sigma)
{
	image S = make_image(im.w, im.h, 3);
	image gx=make_filter(GX,0),gy=make_filter(GY,0);
	image Ix=convolve_image(im,gx,0),Iy=convolve_image(im,gy,0);
	FIMG(gx);FIMG(gy);
	
	int i,sz=im.w*im.h;
	for(i=0;i<sz;++i){
		S.data[i]=Ix.data[i]*Ix.data[i];
		S.data[sz+i]=Iy.data[i]*Iy.data[i];
		S.data[2*sz+i]=Ix.data[i]*Iy.data[i];
	}
	FIMG(Ix);FIMG(Iy);
	gx=smooth_image(S,sigma);
	FIMG(S);	
    return gx;
}

// Estimate the cornerness of each pixel given a structure matrix S.
// image S: structure matrix for an image.
// returns: a response map of cornerness calculations.
image cornerness_response(image S)
{
    image R = make_image(S.w, S.h, 1);
    // TODO: fill in R, "cornerness" for each pixel using the structure matrix.
    // We'll use formulation det(S) - alpha * trace(S)^2, alpha = .06.
	int i=0,sz=S.w*S.h;
	for(;i<sz;++i){
		R.data[i]=S.data[i]*S.data[sz+i]-pow(S.data[2*sz+i],2)-.06*pow(S.data[i]+S.data[sz+i],2);
	}
    return R;
}

// Perform non-max supression on an image of feature responses.
// image im: 1-channel image of feature responses.
// int w: distance to look for larger responses.
// returns: image with only local-maxima responses within w pixels.
image nms_image(image im, int w)
{
    image r = copy_image(im);
    // TODO: perform NMS on the response map.
    // for every pixel in the image:
    //     for neighbors within w:
    //         if neighbor response greater than pixel response:
    //             set response to be very low (I use -999999 [why not 0??])
	int i,j,k,l,m;
	float res;
	for(k=0;k<im.c;++k)for(i=0;i<im.w;++i)for(j=0;j<im.h;){
		res=get_pixel(r,i,j,k);
		for(l=-w;l<=w;++l)
		{
			if(l==0)continue;
			for(m=-w;m<=w;++m){
				if(m==0)continue;
				if(get_pixel(im,i+l,j+m,k)>res){
					set_pixel(r,i,j,k,-999999);
					goto here;
				}
			}
		}
		here:++j;
	}
    return r;
}

// Perform harris corner detection and extract features from the corners.
// image im: input image.
// float sigma: std. dev for harris.
// float thresh: threshold for cornerness.
// int nms: distance to look for local-maxes in response map.
// int *n: pointer to number of corners detected, should fill in.
// returns: array of descriptors of the corners in the image.
descriptor *harris_corner_detector(image im, float sigma, float thresh, int nms, int *n)
{
    // Calculate structure matrix
    image S = structure_matrix(im, sigma);

    // Estimate cornerness
    image R = cornerness_response(S);

    // Run NMS on the responses
    image Rnms = nms_image(R, nms);
	FIMG(S);
    FIMG(R);

    //TODO: count number of responses over threshold
    int i,count = 0; // change this
	for(i=0;i<Rnms.w*Rnms.h;++i)if(Rnms.data[i]>thresh)++count;
    
    *n = count; // <- set *n equal to number of corners in image.
    descriptor *d = calloc(count, sizeof(descriptor));
    //TODO: fill in array *d with descriptors of corners, use describe_index.
	count=0;
	S=sobel(im);
	for(i=0;i<Rnms.w*Rnms.h;++i)if(Rnms.data[i]>thresh)d[count++]=describe_index(S,i);
    FIMG(S);
    FIMG(Rnms);
    return d;
}

// Find and draw corners on an image.
// image im: input image.
// float sigma: std. dev for harris.
// float thresh: threshold for cornerness.
// int nms: distance to look for local-maxes in response map.
void detect_and_draw_corners(image im, float sigma, float thresh, int nms)
{
    int n = 0;
    descriptor *d = harris_corner_detector(im, sigma, thresh, nms, &n);
    mark_corners(im, d, n);
}
