#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "image.h"

// Draws a line on an image with color corresponding to the direction of line
// image im: image to draw line on
// float x, y: starting point of line
// float dx, dy: vector corresponding to line angle and magnitude
void draw_line(image im, float x, float y, float dx, float dy)
{
    assert(im.c == 3);
    float angle = 6*(atan2(dy, dx) / TWOPI + .5);
    int index = floor(angle);
    float f = angle - index;
    float r, g, b;
    if(index == 0){
        r = 1; g = f; b = 0;
    } else if(index == 1){
        r = 1-f; g = 1; b = 0;
    } else if(index == 2){
        r = 0; g = 1; b = f;
    } else if(index == 3){
        r = 0; g = 1-f; b = 1;
    } else if(index == 4){
        r = f; g = 0; b = 1;
    } else {
        r = 1; g = 0; b = 1-f;
    }
    float i;
    float d = sqrt(dx*dx + dy*dy);
    for(i = 0; i < d; i += 1){
        int xi = x + dx*i/d;
        int yi = y + dy*i/d;
        set_pixel(im, xi, yi, 0, r);
        set_pixel(im, xi, yi, 1, g);
        set_pixel(im, xi, yi, 2, b);
    }
}

// Make an integral image or summed area table from an image
// image im: image to process
// returns: image I such that I[x,y] = sum{i<=x, j<=y}(im[i,j])
image make_integral_image(image im,int s)
{
    image integ = make_image(im.w+s, im.h+s, im.c);
    int i,j,k,hs=s>>1;
    for (k = 0; k < integ.c; ++k)for (j = 0; j < integ.h; ++j)for (i = 0; i < integ.w; ++i) {
		set_pixel(integ,i,j,k,
			(i? get_pixel(integ,i-1,j,k) : 0)+
			(j? get_pixel(integ,i,j-1,k) : 0)-
			(i&&j ? get_pixel(integ,i-1,j-1,k) : 0)+
			get_pixel(im,i-hs,j-hs,k));
    }
    return integ;
}

// Apply a box filter to an image using an integral image for speed
// image im: image to smooth
// int s: window size for box filter
// returns: smoothed image
image box_filter_image(image im, int s)
{
    int i,j,k;
	image integ=make_integral_image(im,s);
    image S = make_image(im.w, im.h, im.c);
    // TODO: fill in S using the integral image.
	float f=1.0/(s*s);
	for (k = 0; k < S.c; ++k)for (j = 0; j < S.h; ++j)for (i = 0; i < S.w; ++i){
		set_pixel(S,i,j,k,
			f*(get_pixel(integ,i+s,j+s,k)+get_pixel(integ,i,j,k)-get_pixel(integ,i+s,j,k)-get_pixel(integ,i,j+s,k)));
	}
	FIMG(integ);
    return S;
}

// Calculate the time-structure matrix of an image pair.
// image im: the input image.
// image prev: the previous image in sequence.
// int s: window size for smoothing.
// returns: structure matrix. 1st channel is Ix^2, 2nd channel is Iy^2,
//          3rd channel is IxIy, 4th channel is IxIt, 5th channel is IyIt.
image time_structure_matrix(image im, image prev, int s)
{
    int i,n=im.w*im.h;
    int converted = 0;
    if(im.c == 3){
        converted = 1;
        im = rgb_to_grayscale(im);
        prev = rgb_to_grayscale(prev);
    }
    // TODO: calculate gradients, structure components, and smooth them
	image S=make_image(im.w,im.h,5);
	if(converted){
		image gx= make_filter(GX,0);
		image gy= make_filter(GY,0);
		image Ix = convolve_image(im, gx, 0);
		image Iy = convolve_image(im, gy, 0);
		image It = add_image(im,prev,0);
		for(i=0;i<n;++i){
			S.data[i]=pow(Ix.data[i],2);
			S.data[i+n]=pow(Iy.data[i],2);
			S.data[i+2*n]=Ix.data[i]*Iy.data[i];
			S.data[i+3*n]=Ix.data[i]*It.data[i];
			S.data[i+4*n]=Iy.data[i]*It.data[i];
		}
        FIMG(im); FIMG(prev);FIMG(gx);FIMG(gy);
		FIMG(Ix);FIMG(Iy);FIMG(It);
		It=box_filter_image(S,s);
		FIMG(S);
		S=It;
    }
    return S;
}

// Calculate the velocity given a structure image
// image S: time-structure image
// int stride: only calculate subset of pixels for speed
image velocity_image(image S, int stride)
{
    image v = make_image(S.w/stride, S.h/stride, 3);
    int i, j;
    matrix M = make_matrix(2,2),Mt = make_matrix(2,1);
    for(j = (stride-1)/2; j < S.h; j += stride){
        for(i = (stride-1)/2; i < S.w; i += stride){
            float Ixx = S.data[i + S.w*j + 0*S.w*S.h];
            float Iyy = S.data[i + S.w*j + 1*S.w*S.h];
            float Ixy = S.data[i + S.w*j + 2*S.w*S.h];
            float Ixt = S.data[i + S.w*j + 3*S.w*S.h];
            float Iyt = S.data[i + S.w*j + 4*S.w*S.h];
            // TODO: calculate vx and vy using the flow equation
			float vx = 0,vy = 0;
			M.data[0] = Ixx;
            M.data[1] = Ixy; 
            M.data[2] = Ixy; 
            M.data[3] = Iyy;

            Mt.data[0] = -Ixt; 
            Mt.data[1] = -Iyt; 
			matrix Minv=matrix_invert(M);
			if(Minv.rows>0){
				matrix Vel=matmul(Minv,Mt);
				vx=Vel.data[0];
				vy=Vel.data[1];
				FM(Vel);
			}         
            set_pixel(v, i/stride, j/stride, 0, vx);
            set_pixel(v, i/stride, j/stride, 1, vy);
			FM(Minv);
        }
    }
    FM(M);FM(Mt);
    return v;
}

// Draw lines on an image given the velocity
// image im: image to draw on
// image v: velocity of each pixel
// float scale: scalar to multiply velocity by for drawing
void draw_flow(image im, image v, float scale)
{
    int stride = im.w / v.w;
    int i,j;
    for (j = (stride-1)/2; j < im.h; j += stride) {
        for (i = (stride-1)/2; i < im.w; i += stride) {
            float dx = scale*get_pixel(v, i/stride, j/stride, 0);
            float dy = scale*get_pixel(v, i/stride, j/stride, 1);
            if(fabs(dx) > im.w) dx = 0;
            if(fabs(dy) > im.h) dy = 0;
            draw_line(im, i, j, dx, dy);
        }
    }
}


// Constrain the absolute value of each image pixel
// image im: image to constrain
// float v: each pixel will be in range [-v, v]
void constrain_image(image im, float v)
{
    int i;
    for(i = 0; i < im.w*im.h*im.c; ++i){
        if (im.data[i] < -v) im.data[i] = -v;
        if (im.data[i] >  v) im.data[i] =  v;
    }
}

// Calculate the optical flow between two images
// image im: current image
// image prev: previous image
// int smooth: amount to smooth structure matrix by
// int stride: downsampling for velocity matrix
// returns: velocity matrix
image optical_flow_images(image im, image prev, int smooth, int stride)
{
    image S = time_structure_matrix(im, prev, smooth);   
    image v = velocity_image(S, stride);
    constrain_image(v, 6);
    image vs = smooth_image(v, 2);
    FIMG(v);
    FIMG(S);
    return vs;
}

// Run optical flow demo on webcam
// int smooth: amount to smooth structure matrix by
// int stride: downsampling for velocity matrix
// int div: downsampling factor for images from webcam
void optical_flow_webcam(int smooth, int stride, int div,char *f)
{
#ifdef OPENCV
    void * cap;
    cap = open_video_stream(f?f:0, 0, 1280, 720, 30);
    image prev = get_image_from_stream(cap);
    image prev_c = nn_resize(prev, prev.w/div, prev.h/div);
    image im = get_image_from_stream(cap);
    image im_c = nn_resize(im, im.w/div, im.h/div);
    while(im.data){
        image copy = copy_image(im);
        image v = optical_flow_images(im_c, prev_c, smooth, stride);
        draw_flow(copy, v, smooth*div);
        int key = show_image(copy, "flow", 5);
        FIMG(v);
        FIMG(copy);
        FIMG(prev);
        FIMG(prev_c);
        prev = im;
        prev_c = im_c;
        if(key != -1) {
            key = key % 256;
            printf("%d\n", key);
            if (key == 27) break;
        }
        im = get_image_from_stream(cap);
        im_c = nn_resize(im, im.w/div, im.h/div);
    }
#else
    fprintf(stderr, "Must compile with OpenCV\n");
#endif
}
