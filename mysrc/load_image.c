// You probably don't want to edit this file
#include <stdio.h>
#include <stdlib.h>

#include "image.h"

image make_empty_image(int w, int h, int c)
{
    image out;
    out.data = 0;
    out.h = h;
    out.w = w;
    out.c = c;
    return out;
}

image make_image(int w, int h, int c)
{
    image out = make_empty_image(w,h,c);
    out.data = calloc(h*w*c, sizeof(float));
    return out;
}
image float_to_image(float *data, int w, int h, int c)
{
    image out = {0};
    out.data = data;
    out.w = w;
    out.h = h;
    out.c = c;
    return out;
}
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

void save_image_options(image im, const char *name, IMAGE_TYPE f, int quality)
{
    char buff[256];
    //sprintf(buff, "%s (%d)", name, windows);
    if(f == PNG)       sprintf(buff, "%s.png", name);
    else if (f == BMP) sprintf(buff, "%s.bmp", name);
    else if (f == TGA) sprintf(buff, "%s.tga", name);
    else if (f == JPG) sprintf(buff, "%s.jpg", name);
    else               sprintf(buff, "%s.png", name);
    unsigned char *data = calloc(im.w*im.h*im.c, sizeof(char));
    int i,k;
    for(k = 0; k < im.c; ++k){
        for(i = 0; i < im.w*im.h; ++i){
            data[i*im.c+k] = (unsigned char) (255*im.data[i + k*im.w*im.h]);
        }
    }
    int success = 0;
    if(f == PNG)       success = stbi_write_png(buff, im.w, im.h, im.c, data, im.w*im.c);
    else if (f == BMP) success = stbi_write_bmp(buff, im.w, im.h, im.c, data);
    else if (f == TGA) success = stbi_write_tga(buff, im.w, im.h, im.c, data);
    else if (f == JPG) success = stbi_write_jpg(buff, im.w, im.h, im.c, data, quality);
    free(data);
    if(!success) fprintf(stderr, "Failed to write image %s\n", buff);
}

void save_png(image im, const char *name)
{
    save_image_options(im, name,PNG, 80);
}

void save_image(image im, const char *name)
{
    save_image_options(im, name, JPG, 80);
}

// 
// Load an image using stb
// channels = [0..4]
// channels > 0 forces the image to have that many channels
//
image load_image_stb(char *filename, int channels)
{
    int w, h, c;
    unsigned char *data = stbi_load(filename, &w, &h, &c, channels);
    if (!data) {
        fprintf(stderr, "Cannot load image \"%s\"\nSTB Reason: %s\n",
            filename, stbi_failure_reason());
        exit(0);
    }
    if (channels) c = channels;
    int i,j,k;
    image im = make_image(w, h, c);
    for(k = 0; k < c; ++k){
        for(j = 0; j < h; ++j){
            for(i = 0; i < w; ++i){
                int dst_index = i + w*j + w*h*k;
                int src_index = k + c*i + c*w*j;
                im.data[dst_index] = (float)data[src_index]/255.;
            }
        }
    }
    //We don't like alpha channels, #YOLO
    if(im.c == 4) im.c = 3;
    free(data);
    return im;
}

image load_image(char *filename)
{
    image out = load_image_stb(filename, 0);
    return out;
}

void save_image_binary(image im, const char *fname)
{
    FILE *fp = fopen(fname, "wb");
    fwrite(&im.w, sizeof(int), 1, fp);
    fwrite(&im.h, sizeof(int), 1, fp);
    fwrite(&im.c, sizeof(int), 1, fp);
    fwrite(im.data, sizeof(float), im.w*im.h*im.c, fp);
    fclose(fp);
}

image load_image_binary(const char *fname)
{
    int w = 0;
    int h = 0;
    int c = 0;
    FILE *fp = fopen(fname, "rb");
    fread(&w, sizeof(int), 1, fp);
    fread(&h, sizeof(int), 1, fp);
    fread(&c, sizeof(int), 1, fp);
    image im = make_image(w,h,c);
    fread(im.data, sizeof(float), im.w*im.h*im.c, fp);
    return im;
}

void free_image(image *im)
{
	if(im->data){
		free(im->data);im->data=NULL;
	}
}

