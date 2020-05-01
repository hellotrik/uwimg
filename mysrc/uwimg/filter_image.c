#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "image.h"
//----------filter---------------
void l1_normalize(image im)
{
    int i,j=im.w*im.h*im.c;
	float v=0;
	for(i=0;i<j;++i)v+=im.data[i];
	if(v!=0){
		v=1/v;
		for(i=0;i<j;++i)im.data[i]*=v;
	}
}

image convolve_image(image im, image filter, int preserve)
{
    image g;
	int i,j,k,l,t,n,i1,j1,k1;
	float v;
	if(preserve==1){
		g=make_image(im.w,im.h,im.c);
		n=filter.c>>1;
	}
	else {
		g=make_image(im.w,im.h,1);
		n=0;
	}
	l=filter.w>>1;
	t=filter.h>>1;
	
	for(k=0;k<(preserve==1?im.c:1);++k)for(i=0;i<im.w;++i)for(j=0;j<im.h;++j){
		v=0;
		for(k1=0;k1<filter.c;++k1)for(i1=0;i1<filter.w;++i1)for(j1=0;j1<filter.h;++j1){
				if(filter.c==1&&preserve!=1){
					int k2;
					for(k2=0;k2<im.c;++k2)
					v+=get_pixel(im,i+i1-l,j+j1-t,k+k2+k1-n)*get_pixel(filter,i1,j1,k1);
				}else
					v+=get_pixel(im,i+i1-l,j+j1-t,k+k1-n)*get_pixel(filter,i1,j1,k1);
			}
		set_pixel(g,i,j,k,v);
	}
	return g;
}

image make_filter(KER ker,float w_sigma)
{
	int i,j,hw,hf;
	float v;
	image g;
	switch(ker){
		case BOX:
			j=w_sigma*w_sigma;
			v=1./j;
			g=make_image(w_sigma,w_sigma,1);
			for(i=0;i<j;++i)g.data[i]=v;
			break;
		case HPASS:
			g=make_image(3,3,1);
			g.data[1]=-1;
			g.data[3]=-1;
			g.data[4]=4;
			g.data[5]=-1;
			g.data[7]=-1;
			break;
		case SHARPEN:
			g=make_image(3,3,1);
			g.data[1]=-1;
			g.data[3]=-1;
			g.data[4]=5;
			g.data[5]=-1;
			g.data[7]=-1;
			break;
		case EMBOSS:
			g=make_image(3,3,1);
			g.data[0]=-2;
			g.data[1]=-1;
			g.data[3]=-1;
			g.data[4]=1;
			g.data[5]=1;
			g.data[7]=1;
			g.data[8]=2;
			break;
		case GAUSS:
			hw=6*w_sigma+1;
			if((hw&1)==0)++hw;
			hf=hw>>1;
			g=make_image(hw,hw,1);
			float sig2=w_sigma*w_sigma;
			float twopisig2=1/(TWOPI*sig2);
			float sig2two=1/(2*sig2);
			for(i=0;i<hw;++i)for(j=0;j<hw;++j)set_pixel(g,i,j,0,exp(-((i-hf)*(i-hf)+(j-hf)*(j-hf))*sig2two)*twopisig2);
			l1_normalize(g);
			break;
		case GX:
			g=make_image(3,3,1);
			g.data[0]=-1;
			g.data[2]=1;
			g.data[3]=-2;
			g.data[5]=2;
			g.data[6]=-1;
			g.data[8]=1;
			break;
		case GY:
			g=make_image(3,3,1);
			g.data[0]=-1;
			g.data[1]=-2;
			g.data[2]=-1;
			g.data[6]=1;
			g.data[7]=2;
			g.data[8]=1;
			break;
	}
	return g;
}


void feature_normalize(image im)
{
    float mi=im.data[0],ma=im.data[0],delta;
	int i;
	for(i=1;i<im.w*im.h*im.c;++i){
		if(mi>im.data[i])mi=im.data[i];
		if(ma<im.data[i])ma=im.data[i];
	}
	delta=ma-mi;
	if(delta!=0){
		delta=1/delta;
		for(i=0;i<im.w*im.h*im.c;++i)
			im.data[i]=(im.data[i]-mi)*delta;
	}
}
void threshold_image(image im, float thresh){
	int i;
	for(i=0;i<im.w*im.h*im.c;++i)im.data[i]=im.data[i]>thresh?1:0;
}
image *sobel_image(image im)
{
    image gx=make_filter(GX,0),gy=make_filter(GY,0);
	image gxx=convolve_image(im,gx,0),gyy=convolve_image(im,gy,0);
	FIMG(gx);FIMG(gy);
	
	image *gt=calloc(2, sizeof(image));
	gt[0]=make_image(im.w,im.h,1);
	gt[1]=make_image(im.w,im.h,1);
	int i;
	for(i=0;i<im.w*im.h;++i){
		gt[0].data[i]=sqrt(gxx.data[i]*gxx.data[i]+gyy.data[i]*gyy.data[i]);
		gt[1].data[i]=atan2(gyy.data[i],gxx.data[i]);
	}
	
	FIMG(gxx);	
	FIMG(gyy);
    return gt;
}
image sobel(image im)
{
	image gx=make_filter(GX,0),gy=make_filter(GY,0);
	image gxx=convolve_image(im,gx,0),gyy=convolve_image(im,gy,0);
	FIMG(gx);FIMG(gy);
	image gt=make_image(im.w,im.h,3);
	int i,n=im.w*im.h,n2=sizeof(float)*n;
	for(i=0;i<n;++i){
		gt.data[i]=atan2(gyy.data[i],gxx.data[i]);
	}
	memcpy(gt.data+n,gxx.data,n2);
	memcpy(gt.data+2*n,gyy.data,n2);
	FIMG(gxx);	
	FIMG(gyy);
    return gt;
}

image colorize_sobel(image im)
{
    image *sobel = sobel_image(im);
    feature_normalize(sobel[0]);
    feature_normalize(sobel[1]);
    image res = make_image(im.w, im.h, im.c);
	int n=im.w*im.h;
	int size=sizeof(float)*n;
    memcpy(res.data,sobel[1].data,size);
    memcpy(res.data+n,sobel[0].data,size);
    memcpy(res.data+2*n,sobel[0].data,size);
    	
    FIMG(sobel[1]);
    FIMG(sobel[0]);
    free(sobel);
    hsv_to_rgb(res);
    return res;
}
//---------------process_image--------------

float polate(image im, float x, float y, int c,int nn)
{
	if(nn)return get_pixel(im,x+0.5,y+0.5,c);
	//bilinear
	float d1=x-(int)x,d2=1-d1,d3=y-(int)y,d4=1-d3;
	return 	get_pixel(im,x,y,c)*d2*d4+
			get_pixel(im,x+1,y,c)*d1*d4+
			get_pixel(im,x,y+1,c)*d2*d3+
			get_pixel(im,x+1,y+1,c)*d1*d3;		
}

image resize(image im, int w, int h,int nn)
{
    // TODO Fill in (also fix that first line)
	image g=make_image(w,h,im.c);
	int i,j,k;
	float a,a1,b,b1,m,n;
	//ax+b=y a*-.5+b=-.5 a*w+b=im.w
	a=im.w/(float)w;
	a1=im.h/(float)h;
	b=(a-1)/2;
	b1=(a1-1)/2;
	for(i=0;i<w;++i){
		m=a*i+b;
		for(j=0;j<h;++j){
			n=j*a1+b1;
			for(k=0;k<im.c;++k){
				set_pixel(g,i,j,k,polate(im,m,n,k,nn));
			}
		}
	}
    return g;
}

float get_pixel(image im, int x, int y, int c)
{
    // TODO Fill this in
	x=x<0?0:x>=im.w?im.w-1:x;
	y=y<0?0:y>=im.h?im.h-1:y;
	c=c<0?0:c>=im.c?im.c-1:c;
    return im.data[im.w*(c*im.h+y)+x];
}

void set_pixel(image im, int x, int y, int c, float v)
{
    // TODO Fill this in
	if(x<0||x>=im.w||y<0||y>=im.h||c<0||c>=im.c) return;
	im.data[im.w*(c*im.h+y)+x]=v;
}

image copy_image(image im)
{
    image copy = make_image(im.w, im.h, im.c);
	memcpy(copy.data, im.data, sizeof(float) * im.w * im.h * im.c);
    return copy;
}

image rgb_to_grayscale(image im)
{
    assert(im.c == 3);
    image gray = make_image(im.w, im.h, 1);
	int i,k=im.h*im.w;
	for(i=0;i<k;++i)
	gray.data[i]=im.data[i]*0.299+im.data[k+i]*0.587+im.data[2*k+i]*0.114;
    return gray;
}

void shift_image(image im, int c, float v)
{
    c=c<0?0:c>=im.c?im.c-1:c;
	int i;
	int cc=c*im.w*im.h;
	for(i=cc;i<cc+im.w*im.h;++i)im.data[i]+=v;
}
void scale_image(image im, int c, float v)
{
	c=c<0?0:c>=im.c?im.c-1:c;
	int i;
	int cc=c*im.w*im.h;
	for(i=cc;i<cc+im.w*im.h;++i)im.data[i]*=v;
}
image add_image(image a, image b,int sign)
{
	image g =copy_image(a);
	int i,j,k;
	for(i=0;i<a.w;++i)for(j=0;j<a.h;++j)for(k=0;k<a.c;++k)g.data[(k*a.h+j)*a.w+i]+=(i>=b.w||j>=b.h||k>=b.c)?0:sign?get_pixel(b,i,j,k):-get_pixel(b,i,j,k);
	return g;
}

image get_channel(image im, int c){
	c=c<0?0:c>=im.c?im.c-1:c;
	image r=make_image(im.w,im.h,1);
	int n=im.w*im.h;
	memcpy(r.data,im.data+n*c,sizeof(float)*n);
	return r;
}

void clamp_image(image im)
{
    int i;
	for(i=0;i<im.w*im.h*im.c;++i)if(im.data[i]>1)im.data[i]=1;else if(im.data[i]<0)im.data[i]=0;
}
// These might be handy
float three_way_max(float a, float b, float c)
{
    return (a > b) ? ( (a > c) ? a : c) : ( (b > c) ? b : c) ;
}

float three_way_min(float a, float b, float c)
{
    return (a < b) ? ( (a < c) ? a : c) : ( (b < c) ? b : c) ;
}

void rgb_to_hsv(image im)
{
    int i,k=im.w*im.h;
	float h,h1,s,v,c;
	float r,g,b;
	for(i=0;i<k;++i){
		r=im.data[i];
		g=im.data[k+i];
		b=im.data[2*k+i];
		v=three_way_max(r,g,b);
		c=v-three_way_min(r,g,b);
		s=v==0?0:c/v;
		h1=c==0?0:r==v?(g-b)/c:
			g==v?(b-r)/c+2:
			(r-g)/c+4;
		h=h1<0?h1/6+1:h1/6;
		im.data[i]=h;
		im.data[i+k]=s;
		im.data[i+2*k]=v;
	}
}

void hsv_to_rgb(image im)
{
    int i,hi,k=im.w*im.h;
	float h,h1,s,v,c,f;
	float r,g,b;
	float p,q,t;
	for(i=0;i<k;++i){
		h=im.data[i];
		s=im.data[k+i];
		v=im.data[2*k+i];
		c=v*s;
		h1=6*h;
		hi=(int)h1;
		f=h1-hi;
		p= v-c;
		q= v*(1-f*s);
		t=v*(1-(1-f)*s);
		switch(hi){
		case 0:
			r=v;g=t;b=p;break;
		case 1:
			r=q;g=v;b=p;break;
		case 2:
			r=p;g=v;b=t;break;
		case 3:
			r=p;g=q;b=v;break;
		case 4:
			r=t;g=p;b=v;break;
		default:
			r=v;g=p;b=q;break;			
		}
		im.data[i]=r;
		im.data[i+k]=g;
		im.data[i+2*k]=b;
	}
}
