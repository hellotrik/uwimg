#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include "uwnet.h"
#include "matrix.h"
#include "image.h"
#include "test.h"
#include "args.h"

static double f64rand()
{
	static int randbit = 0;
	if (!randbit)
	{
		srand((unsigned)time(NULL));
		for (int i = RAND_MAX; i; i >>= 1, ++randbit);
	}
	unsigned long long lvalue = 0x4000000000000000L;
	int i = 52 - randbit;
	for (; i > 0; i -= randbit)
		lvalue |= (unsigned long long)rand() << i;
	lvalue |= (unsigned long long)rand() >> -i;
	double *ret=(double *)&lvalue;
	return *ret-3;
}

static float avg_diff(image a, image b)
{
    float diff = 0;
    int i;
    for(i = 0; i < a.w*a.h*a.c; ++i){
        diff += b.data[i] - a.data[i];
    }
    return diff/(a.w*a.h*a.c);
}

static image center_crop(image im)
{
    image c = make_image(im.w/2, im.h/2, im.c);
    int i, j, k;
    for(k = 0; k < im.c; ++k){
        for(j = 0; j < im.h/2; ++j){
            for(i = 0; i < im.w/2; ++i){
                set_pixel(c, i, j, k, get_pixel(im, i+im.w/4, j+im.h/4, k));
            }
        }
    }
    return c;
}

static void feature_normalize2(image im)
{
    int i;
    float min = im.data[0];
    float max = im.data[0];
    for(i = 0; i < im.w*im.h*im.c; ++i){
        if(im.data[i] > max) max = im.data[i];
        if(im.data[i] < min) min = im.data[i];
    }
    for(i = 0; i < im.w*im.h*im.c; ++i){
        im.data[i] = (im.data[i] - min)/(max-min);
    }
}

int tests_total = 0;
int tests_fail = 0;
int within_eps(float a, float b,float eps){
    return a-eps<b && b<a+eps;
}

int same_point(point p, point q, float eps)
{
    return within_eps(p.x, q.x, eps) && within_eps(p.y, q.y, eps);
}

int same_matrix(matrix a, matrix b)
{
    int i;
    if(a.rows != b.rows || a.cols != b.cols) {
        printf ("first matrix: %dx%d, second matrix:%dx%d\n", a.rows, a.cols, b.rows, b.cols);
        return 0;
    }
    for(i = 0; i < a.rows*a.cols; ++i){
        if(!within_eps(a.data[i], b.data[i],EPS*2)) {
            printf("differs at %d, %f vs %f\n", i, a.data[i], b.data[i]);
            return 0;
        }
    }
    return 1;
}

int same_image(image a, image b, float eps)
{
    int i;
    if(a.w != b.w || a.h != b.h || a.c != b.c) {
        //printf("Expected %d x %d x %d image, got %d x %d x %d\n", b.w, b.h, b.c, a.w, a.h, a.c);
        return 0;
    }
    for(i = 0; i < a.w*a.h*a.c; ++i){
        float thresh = (fabs(b.data[i]) + fabs(a.data[i])) * eps / 2;
        if (thresh > eps) eps = thresh;
        if(!within_eps(a.data[i], b.data[i], eps)) 
        {
            printf("The %d value should be %f, but it is %f! \n",i, b.data[i], a.data[i]);
            return 0;
        }
    }
    return 1;
}

double what_time_is_it_now()
{
	return clock()/CLOCKS_PER_SEC;
}
//img
void test_get_pixel(){
    image im = load_image("data/dots.png");
    // Test within image
    TEST(within_eps(0, get_pixel(im, 0,0,0), EPS));
    TEST(within_eps(1, get_pixel(im, 1,0,1), EPS));
    TEST(within_eps(0, get_pixel(im, 2,0,1), EPS));

    // Test padding
    TEST(within_eps(1, get_pixel(im, 0,3,1), EPS));
    TEST(within_eps(1, get_pixel(im, 7,8,0), EPS));
    TEST(within_eps(0, get_pixel(im, 7,8,1), EPS));
    TEST(within_eps(1, get_pixel(im, 7,8,2), EPS));
    FIMG(im);
}

void test_set_pixel(){
    image gt = load_image("data/dots.png");
    image d = make_image(4,2,3);
    set_pixel(d, 0,0,0,0); set_pixel(d, 0,0,1,0); set_pixel(d, 0,0,2,0); 
    set_pixel(d, 1,0,0,1); set_pixel(d, 1,0,1,1); set_pixel(d, 1,0,2,1); 
    set_pixel(d, 2,0,0,1); set_pixel(d, 2,0,1,0); set_pixel(d, 2,0,2,0); 
    set_pixel(d, 3,0,0,1); set_pixel(d, 3,0,1,1); set_pixel(d, 3,0,2,0); 

    set_pixel(d, 0,1,0,0); set_pixel(d, 0,1,1,1); set_pixel(d, 0,1,2,0); 
    set_pixel(d, 1,1,0,0); set_pixel(d, 1,1,1,1); set_pixel(d, 1,1,2,1); 
    set_pixel(d, 2,1,0,0); set_pixel(d, 2,1,1,0); set_pixel(d, 2,1,2,1); 
    set_pixel(d, 3,1,0,1); set_pixel(d, 3,1,1,0); set_pixel(d, 3,1,2,1); 

    // Test images are same
    TEST(same_image(d, gt, EPS));
    FIMG(gt);
    FIMG(d);
}

void test_grayscale()
{
    image im = load_image("data/colorbar.png");
    image gray = rgb_to_grayscale(im);
    image gt = load_image("figs/gray.png");
    TEST(same_image(gray, gt, EPS));
    FIMG(im);
    FIMG(gray);
    FIMG(gt);
}

void test_copy()
{
    image gt = load_image("data/dog.jpg");
    image c = copy_image(gt);
    TEST(same_image(c, gt, EPS));
    FIMG(gt);
    FIMG(c);
}

void test_clamp()
{
    image im = load_image("data/dog.jpg");
    image c = copy_image(im);
    set_pixel(im, 10, 5, 0, -1);
    set_pixel(im, 15, 15, 1, 1.001);
    set_pixel(im, 130, 105, 2, -0.01);
    set_pixel(im, im.w-1, im.h-1, im.c-1, -.01);

    set_pixel(c, 10, 5, 0, 0);
    set_pixel(c, 15, 15, 1, 1);
    set_pixel(c, 130, 105, 2, 0);
    set_pixel(c, im.w-1, im.h-1, im.c-1, 0);
    clamp_image(im);
    TEST(same_image(c, im, EPS));
    FIMG(im);
    FIMG(c);
}

void test_shift()
{
    image im = load_image("data/dog.jpg");
    image c = copy_image(im);
    shift_image(c, 1, .1);
    TEST(within_eps(c.data[0], im.data[0], EPS));
    TEST(within_eps(c.data[im.w*im.h + 13], im.data[im.w*im.h+13] + .1, EPS));
    TEST(within_eps(c.data[2*im.w*im.h + 72], im.data[2*im.w*im.h+72], EPS));
    TEST(within_eps(c.data[im.w*im.h + 47], im.data[im.w*im.h+47] + .1, EPS));
    FIMG(im);
    FIMG(c);
}

void test_rgb_to_hsv()
{
    image im = load_image("data/dog.jpg");
    rgb_to_hsv(im);
    image hsv = load_image("figs/dog.hsv.png");
    TEST(same_image(im, hsv, EPS));
    FIMG(im);
    FIMG(hsv);
}

void test_hsv_to_rgb()
{
    image im = load_image("data/dog.jpg");
    image c = copy_image(im);
    rgb_to_hsv(im);
    hsv_to_rgb(im);
    TEST(same_image(im, c, EPS));
    FIMG(im);
    FIMG(c);
}

void test_nn_interpolate()
{
    image im = load_image("data/dogsmall.jpg");
    TEST(within_eps(polate(im, -.5, -.5, 0,1)  , 0.231373, EPS));
    TEST(within_eps(polate(im, -.5, .5, 1,1)   , 0.239216, EPS));
    TEST(within_eps(polate(im, .499, .5, 2,1)  , 0.207843, EPS));
    TEST(within_eps(polate(im, 14.2, 15.9, 1,1), 0.690196, EPS));
    FIMG(im);
}

void test_bl_interpolate()
{
    image im = load_image("data/dogsmall.jpg");
    TEST(within_eps(polate(im, -.5, -.5, 0,0)  , 0.231373, EPS));
    TEST(within_eps(polate(im, -.5, .5, 1,0)   , 0.237255, EPS));
    TEST(within_eps(polate(im, .499, .5, 2,0)  , 0.206861, EPS));
    TEST(within_eps(polate(im, 14.2, 15.9, 1,0), 0.678588, EPS));
}



void test_nn_resize()
{
    image im = load_image("data/dogsmall.jpg");
    image resized = resize(im, im.w*4, im.h*4,1);
    image gt = load_image("figs/dog4x-nn-for-test.png");
    TEST(same_image(resized, gt, EPS));
    FIMG(im);
    FIMG(resized);
    FIMG(gt);

    image im2 = load_image("data/dog.jpg");
    image resized2 = resize(im2, 713, 467,1);
    image gt2 = load_image("figs/dog-resize-nn.png");
    TEST(same_image(resized2, gt2, EPS));
    FIMG(im2);
    FIMG(resized2);
    FIMG(gt2);
}

void test_bl_resize()
{
    image im = load_image("data/dogsmall.jpg");
    image resized = resize(im, im.w*4, im.h*4,0);
    image gt = load_image("figs/dog4x-bl.png");
    TEST(same_image(resized, gt, EPS));
    FIMG(im);
    FIMG(resized);
    FIMG(gt);

    image im2 = load_image("data/dog.jpg");
    image resized2 = resize(im2, 713, 467,0);
    image gt2 = load_image("figs/dog-resize-bil.png");
    TEST(same_image(resized2, gt2, EPS));
    FIMG(im2);
    FIMG(resized2);
    FIMG(gt2);
}

void test_multiple_resize()
{
    image im = load_image("data/dog.jpg");
    int i;
    for (i = 0; i < 10; ++i){
        image im1 = resize(im, im.w*4, im.h*4,0);
        image im2 = resize(im1, im1.w/4, im1.h/4,0);
        FIMG(im);
        FIMG(im1);
        im = im2;
    }
    image gt = load_image("figs/dog-multipleresize.png");
    TEST(same_image(im, gt, EPS));
    FIMG(im);
    FIMG(gt);
}


void test_highpass_filter(){
    image im = load_image("data/dog.jpg");
    image f = make_filter(HPASS,0);
    image blur = convolve_image(im, f, 0);
    clamp_image(blur);

    
    image gt = load_image("figs/dog-highpass.png");
    TEST(same_image(blur, gt, EPS));
    FIMG(im);
    FIMG(f);
    FIMG(blur);
    FIMG(gt);
}

void test_emboss_filter(){
    image im = load_image("data/dog.jpg");
    image f = make_filter(EMBOSS,0);
    image blur = convolve_image(im, f, 1);
    clamp_image(blur);

    
    image gt = load_image("figs/dog-emboss.png");
    TEST(same_image(blur, gt, EPS));
    FIMG(im);
    FIMG(f);
    FIMG(blur);
    FIMG(gt);
}

void test_sharpen_filter(){
    image im = load_image("data/dog.jpg");
    image f = make_filter(SHARPEN,0);
    image blur = convolve_image(im, f, 1);
    clamp_image(blur);


    image gt = load_image("figs/dog-sharpen.png");
    TEST(same_image(blur, gt, EPS));
    FIMG(im);
    FIMG(f);
    FIMG(blur);
    FIMG(gt);
}

void test_convolution(){
    image im = load_image("data/dog.jpg");
    image f = make_filter(BOX,7);
    image blur = convolve_image(im, f, 1);
    clamp_image(blur);

    image gt = load_image("figs/dog-box7.png");
    TEST(same_image(blur, gt, EPS));
    FIMG(im);
    FIMG(f);
    FIMG(blur);
    FIMG(gt);
}

void test_gaussian_filter(){
    image f = make_filter(GAUSS,7);
    int i;

    for(i = 0; i < f.w * f.h * f.c; ++i){
        f.data[i] *= 100;
    }

    image gt = load_image("figs/gaussian_filter_7.png");
    TEST(same_image(f, gt, EPS));
    FIMG(f);
    FIMG(gt);
}

void test_gaussian_blur(){
    image im = load_image("data/dog.jpg");
    image f = make_filter(GAUSS,2);
    image blur = convolve_image(im, f, 1);
    clamp_image(blur);

    image gt = load_image("figs/dog-gauss2.png");
    TEST(same_image(blur, gt, EPS));
    FIMG(im);
    FIMG(f);
    FIMG(blur);
    FIMG(gt);
}

void test_hybrid_image(){
    image melisa = load_image("data/melisa.png");
    image aria = load_image("data/aria.png");
    image f = make_filter(GAUSS,2);
    image lfreq_m = convolve_image(melisa, f, 1);
    image lfreq_a = convolve_image(aria, f, 1);
    image hfreq_a = add_image(aria , lfreq_a,0);
    image reconstruct = add_image(lfreq_m , hfreq_a,1);
    image gt = load_image("figs/hybrid.png");
    clamp_image(reconstruct);
    TEST(same_image(reconstruct, gt, EPS));
    FIMG(melisa);
    FIMG(aria);
    FIMG(f);
    FIMG(lfreq_m);
    FIMG(lfreq_a);
    FIMG(hfreq_a);
    FIMG(reconstruct);
    FIMG(gt);
}

void test_frequency_image(){
    image im = load_image("data/dog.jpg");
    image f = make_filter(GAUSS,2);
    image lfreq = convolve_image(im, f, 1);
    image hfreq = add_image(im, lfreq,0);
    image reconstruct = add_image(lfreq , hfreq,1);

    image low_freq = load_image("figs/low-frequency.png");
    image high_freq = load_image("figs/high-frequency-clamp.png");

    clamp_image(lfreq);
    clamp_image(hfreq);
    TEST(same_image(lfreq, low_freq, EPS));
    TEST(same_image(hfreq, high_freq, EPS));
    TEST(same_image(reconstruct, im, EPS));
    FIMG(im);
    FIMG(f);
    FIMG(lfreq);
    FIMG(hfreq);
    FIMG(reconstruct);
    FIMG(low_freq);
    FIMG(high_freq);
}

void test_sobel(){
    image im = load_image("data/dog.jpg");
    image *res = sobel_image(im);
    image mag = res[0];
    image theta = res[1];
    feature_normalize2(mag);
    feature_normalize2(theta);

    image gt_mag = load_image("figs/magnitude.png");
    image gt_theta = load_image("figs/theta.png");
    TEST(gt_mag.w == mag.w && gt_theta.w == theta.w);
    TEST(gt_mag.h == mag.h && gt_theta.h == theta.h);
    TEST(gt_mag.c == mag.c && gt_theta.c == theta.c);
    if( gt_mag.w != mag.w || gt_theta.w != theta.w || 
        gt_mag.h != mag.h || gt_theta.h != theta.h || 
        gt_mag.c != mag.c || gt_theta.c != theta.c ) return;
    int i;
    for(i = 0; i < gt_mag.w*gt_mag.h; ++i){
        if(within_eps(gt_mag.data[i], 0, EPS)){
            gt_theta.data[i] = 0;
            theta.data[i] = 0;
        }
        if(within_eps(gt_theta.data[i], 0, EPS) || within_eps(gt_theta.data[i], 1, EPS)){
            gt_theta.data[i] = 0;
            theta.data[i] = 0;
        }
    }

    TEST(same_image(mag, gt_mag, EPS));
    TEST(same_image(theta, gt_theta, EPS));
    FIMG(im);
    FIMG(mag);
    FIMG(theta);
    FIMG(gt_mag);
    FIMG(gt_theta);
    free(res);
}


//matrix1

void test_structure()
{
    image im = load_image("data/dogbw.png");
    image s = structure_matrix(im, 2);
    feature_normalize2(s);
    image gt = load_image("figs/structure.png");
    TEST(same_image(s, gt, EPS));
    FIMG(im);
    FIMG(s);
    FIMG(gt);
}

void test_cornerness()
{
    image im = load_image("data/dogbw.png");
    image s = structure_matrix(im, 2);
    image c = cornerness_response(s);
    feature_normalize2(c);
    image gt = load_image("figs/response.png");
    TEST(same_image(c, gt, EPS));
    FIMG(im);
    FIMG(s);
    FIMG(c);
    FIMG(gt);
}

void test_projection()
{
    matrix H = make_translation_homography(12.4, -3.2);
    TEST(same_point(project_point(H, make_point(0,0)), make_point(12.4, -3.2), EPS));
    FM(H);

    H = make_identity_homography();
    H.data[0] = 1.32;
    H.data[1] = -1.12;
    H.data[2] = 2.52;
    H.data[3] = -.32;
    H.data[4] = -1.2;
    H.data[5] = .52;
    H.data[6] = -3.32;
    H.data[7] = 1.87;
    H.data[8] = .112;
    point p = project_point(H, make_point(3.14, 1.59));
    TEST(same_point(p, make_point(-0.66544, 0.326017), EPS));
    FM(H);
}

void test_compute_homography()
{
    match *m = calloc(4, sizeof(match));
    m[0].p = make_point(0,0);
    m[0].q = make_point(10,10);
    m[1].p = make_point(3,3);
    m[1].q = make_point(13,13);
    m[2].p = make_point(-1.2,-3.4);
    m[2].q = make_point(8.8,6.6);
    m[3].p = make_point(9,10);
    m[3].q = make_point(19,20);
    matrix H = compute_homography(m, 4);
    matrix d10 = make_translation_homography(10, 10);
	print_matrix(H);
    TEST(same_matrix(H, d10));
    FM(H);
    FM(d10);

    m[0].p = make_point(7.2,1.3);
    m[0].q = make_point(10,10.9);
    m[1].p = make_point(3,3);
    m[1].q = make_point(1.3,7.3);
    m[2].p = make_point(-.2,-3.4);
    m[2].q = make_point(.8,2.6);
    m[3].p = make_point(-3.2,2.4);
    m[3].q = make_point(1.5,-4.2);
    H = compute_homography(m, 4);
    matrix Hp = make_identity_homography();
    Hp.data[0] = -0.1328042; Hp.data[1] = -0.2910411; Hp.data[2] = 0.8103200;
    Hp.data[3] = -0.0487439; Hp.data[4] = -1.3077799; Hp.data[5] = 1.4796660;
    Hp.data[6] = -0.0788730; Hp.data[7] = -0.3727209; Hp.data[8] = 1.0000000;
    TEST(same_matrix(H, Hp));
    FM(H);
    FM(Hp);
}


//matrix2


void test_copy_matrix()
{
    matrix a = random_matrix(32, 64, 10);
    matrix c = copy_matrix(a);
    TEST(same_matrix(a,c));
    FM(a);
    FM(c);
}

void test_transpose_matrix()
{
    matrix a = load_matrix("data/test/a.matrix");
    matrix at = load_matrix("data/test/at.matrix");
    matrix atest = transpose_matrix(a);
    matrix aorig = transpose_matrix(atest);
    TEST(same_matrix(at, atest) && same_matrix(a, aorig));
    FM(a);
    FM(at);
    FM(atest);
    FM(aorig);
}

void test_axpy_matrix()
{
    matrix a = load_matrix("data/test/a.matrix");
    matrix y = load_matrix("data/test/y.matrix");
    matrix y1 = load_matrix("data/test/y1.matrix");
    axpy_matrix(2, a, y);
    TEST(same_matrix(y, y1));
    FM(a);
    FM(y);
    FM(y1);
}

void test_matmul()
{
    matrix a = load_matrix("data/test/a.matrix");
    matrix b = load_matrix("data/test/b.matrix");
    matrix c = load_matrix("data/test/c.matrix");
    matrix mul = matmul(a, b);
    TEST(same_matrix(c, mul));
    FM(a);
    FM(b);
    FM(c);
    FM(mul);
}

void test_activate_matrix()
{
    matrix a = load_matrix("data/test/a.matrix");
    matrix truth_alog = load_matrix("data/test/alog.matrix");
    matrix truth_arelu = load_matrix("data/test/arelu.matrix");
    matrix truth_alrelu = load_matrix("data/test/alrelu.matrix");
    matrix truth_asoft = load_matrix("data/test/asoft.matrix");
    matrix alog = copy_matrix(a);
    activate_matrix(alog, LOGISTIC);
    matrix arelu = copy_matrix(a);
    activate_matrix(arelu, RELU);
    matrix alrelu = copy_matrix(a);
    activate_matrix(alrelu, LRELU);
    matrix asoft = copy_matrix(a);
    activate_matrix(asoft, SOFTMAX);
    TEST(same_matrix(truth_alog, alog));
    TEST(same_matrix(truth_arelu, arelu));
    TEST(same_matrix(truth_alrelu, alrelu));
    TEST(same_matrix(truth_asoft, asoft));
    FM(a);
    FM(alog);
    FM(arelu);
    FM(alrelu);
    FM(asoft);
    FM(truth_alog);
    FM(truth_arelu);
    FM(truth_alrelu);
    FM(truth_asoft);
}

void test_gradient_matrix()
{
    matrix a = load_matrix("data/test/a.matrix");
    matrix y = load_matrix("data/test/y.matrix");
    matrix truth_glog = load_matrix("data/test/glog.matrix");
    matrix truth_grelu = load_matrix("data/test/grelu.matrix");
    matrix truth_glrelu = load_matrix("data/test/glrelu.matrix");
    matrix truth_gsoft = load_matrix("data/test/gsoft.matrix");
    matrix glog = copy_matrix(a);
    matrix grelu = copy_matrix(a);
    matrix glrelu = copy_matrix(a);
    matrix gsoft = copy_matrix(a);
    gradient_matrix(y, LOGISTIC, glog);
    gradient_matrix(y, RELU, grelu);
    gradient_matrix(y, LRELU, glrelu);
    gradient_matrix(y, SOFTMAX, gsoft);
    TEST(same_matrix(truth_glog, glog));
    TEST(same_matrix(truth_grelu, grelu));
    TEST(same_matrix(truth_glrelu, glrelu));
    TEST(same_matrix(truth_gsoft, gsoft));
    FM(a);
    FM(glog);
    FM(grelu);
    FM(glrelu);
    FM(gsoft);
    FM(truth_glog);
    FM(truth_grelu);
    FM(truth_glrelu);
    FM(truth_gsoft);
}

void test_connected_layer()
{
    matrix a = load_matrix("data/test/a.matrix");
    matrix b = load_matrix("data/test/b.matrix");
    matrix dw = load_matrix("data/test/dw.matrix");
    matrix db = load_matrix("data/test/db.matrix");
    matrix delta = load_matrix("data/test/delta.matrix");
    matrix prev_delta = load_matrix("data/test/prev_delta.matrix");
    matrix truth_prev_delta = load_matrix("data/test/truth_prev_delta.matrix");
    matrix truth_dw = load_matrix("data/test/truth_dw.matrix");
    matrix truth_db = load_matrix("data/test/truth_db.matrix");
    matrix updated_dw = load_matrix("data/test/updated_dw.matrix");
    matrix updated_db = load_matrix("data/test/updated_db.matrix");
    matrix updated_w = load_matrix("data/test/updated_w.matrix");
    matrix updated_b = load_matrix("data/test/updated_b.matrix");

    matrix bias = load_matrix("data/test/bias.matrix");
    matrix truth_out = load_matrix("data/test/out.matrix");
    layer l = make_connected_layer(64, 16, LRELU);
    l.w = b;
    l.b = bias;
    l.dw = dw;
    l.db = db;
    matrix out = l.forward(l, a);
    TEST(same_matrix(truth_out, out));


    l.delta[0] = delta;
    l.backward(l, prev_delta);
    TEST(same_matrix(truth_prev_delta, prev_delta));
    TEST(same_matrix(truth_dw, l.dw));
    TEST(same_matrix(truth_db, l.db));

    l.update(l, .01, .9, .01);
    TEST(same_matrix(updated_dw, l.dw));
    TEST(same_matrix(updated_db, l.db));
    TEST(same_matrix(updated_w, l.w));
    TEST(same_matrix(updated_b, l.b));

    FM(a);
    FM(b);
    FM(bias);
    FM(out);
    FM(truth_out);
}

void test_matrix_mean()
{
    matrix a = load_matrix("data/test/a.matrix");
    matrix truth_mu_a = load_matrix("data/test/mu_a.matrix");
    matrix truth_mu_a_s = load_matrix("data/test/mu_a_s.matrix");

    matrix mu_a   = mean(a, 1);
    matrix mu_a_s = mean(a, 8);

    TEST(same_matrix(truth_mu_a, mu_a));
    TEST(same_matrix(truth_mu_a_s, mu_a_s));

    FM(a);
    FM(mu_a);
    FM(mu_a_s);
    FM(truth_mu_a);
    FM(truth_mu_a_s);
}

void test_matrix_variance()
{
    matrix a        = load_matrix("data/test/a.matrix");
    matrix mu_a     = load_matrix("data/test/mu_a.matrix");
    matrix mu_a_s   = load_matrix("data/test/mu_a_s.matrix");

    matrix sig_a    =  variance(a, mu_a, 1);
    matrix sig_a_s  =  variance(a, mu_a_s, 8);

    matrix truth_sig_a = load_matrix("data/test/sig_a.matrix");
    matrix truth_sig_a_s = load_matrix("data/test/sig_a_s.matrix");

    TEST(same_matrix(truth_sig_a, sig_a));
    TEST(same_matrix(truth_sig_a_s, sig_a_s));

    FM(a);
    FM(mu_a);
    FM(mu_a_s);
    FM(sig_a);
    FM(sig_a_s);
    FM(truth_sig_a);
    FM(truth_sig_a_s);
}

void test_matrix_normalize()
{
    matrix a        = load_matrix("data/test/a.matrix");
    matrix mu_a     = load_matrix("data/test/mu_a.matrix");
    matrix mu_a_s   = load_matrix("data/test/mu_a_s.matrix");
    matrix sig_a    = load_matrix("data/test/sig_a.matrix");
    matrix sig_a_s  = load_matrix("data/test/sig_a_s.matrix");
    matrix truth_norm_a   = load_matrix("data/test/norm_a.matrix");
    matrix truth_norm_a_s = load_matrix("data/test/norm_a_s.matrix");

    matrix norm_a = normalize(a, mu_a, sig_a, 1);
    matrix norm_a_s = normalize(a, mu_a_s, sig_a_s, 8);

    TEST(same_matrix(truth_norm_a,   norm_a));
    TEST(same_matrix(truth_norm_a_s, norm_a_s));

    FM(a);
    FM(mu_a);
    FM(mu_a_s);
    FM(sig_a);
    FM(sig_a_s);
    FM(norm_a);
    FM(norm_a_s);
    FM(truth_norm_a);
    FM(truth_norm_a_s);
}

void test_matrix_delta_mean()
{
    matrix a        = load_matrix("data/test/a.matrix");
    matrix d        = load_matrix("data/test/y.matrix");
    matrix mu_a     = load_matrix("data/test/mu_a.matrix");
    matrix mu_a_s   = load_matrix("data/test/mu_a_s.matrix");
    matrix sig_a    = load_matrix("data/test/sig_a.matrix");
    matrix sig_a_s  = load_matrix("data/test/sig_a_s.matrix");
    matrix truth_dm       = load_matrix("data/test/dm.matrix");
    matrix truth_dm_s     = load_matrix("data/test/dm_s.matrix");

    matrix dm = delta_mean(d, sig_a, 1);
    matrix dm_s = delta_mean(d, sig_a_s, 8);

    TEST(same_matrix(truth_dm,   dm));
    TEST(same_matrix(truth_dm_s, dm_s));

    FM(a);
    FM(mu_a);
    FM(mu_a_s);
    FM(sig_a);
    FM(sig_a_s);
}

void test_matrix_delta_variance()
{
    matrix a        = load_matrix("data/test/a.matrix");
    matrix d        = load_matrix("data/test/y.matrix");
    matrix mu_a     = load_matrix("data/test/mu_a.matrix");
    matrix mu_a_s   = load_matrix("data/test/mu_a_s.matrix");
    matrix sig_a    = load_matrix("data/test/sig_a.matrix");
    matrix sig_a_s  = load_matrix("data/test/sig_a_s.matrix");
    matrix truth_dv       = load_matrix("data/test/dv.matrix");
    matrix truth_dv_s     = load_matrix("data/test/dv_s.matrix");

    matrix dv = delta_variance(d, a, mu_a, sig_a, 1);
    matrix dv_s = delta_variance(d, a, mu_a_s, sig_a_s, 8);

    TEST(same_matrix(truth_dv,   dv));
    TEST(same_matrix(truth_dv_s, dv_s));

    FM(a);
    FM(mu_a);
    FM(mu_a_s);
    FM(sig_a);
    FM(sig_a_s);
}

void test_matrix_delta_normalize()
{
    matrix a        = load_matrix("data/test/a.matrix");
    matrix d        = load_matrix("data/test/y.matrix");
    matrix mu_a     = load_matrix("data/test/mu_a.matrix");
    matrix mu_a_s   = load_matrix("data/test/mu_a_s.matrix");
    matrix sig_a    = load_matrix("data/test/sig_a.matrix");
    matrix sig_a_s  = load_matrix("data/test/sig_a_s.matrix");
    matrix dm       = load_matrix("data/test/dm.matrix");
    matrix dm_s     = load_matrix("data/test/dm_s.matrix");
    matrix dv       = load_matrix("data/test/dv.matrix");
    matrix dv_s     = load_matrix("data/test/dv_s.matrix");
    matrix truth_dbn      = load_matrix("data/test/dbn.matrix");
    matrix truth_dbn_s    = load_matrix("data/test/dbn_s.matrix");

    matrix dbn = delta_batch_norm(d, dm, dv, mu_a, sig_a, a, 1);
    matrix dbn_s = delta_batch_norm(d, dm_s, dv_s, mu_a_s, sig_a_s, a, 8);

    TEST(same_matrix(truth_dbn,   dbn));
    TEST(same_matrix(truth_dbn_s, dbn_s));

    FM(a);
    FM(mu_a);
    FM(mu_a_s);
    FM(sig_a);
    FM(sig_a_s);
}

void test_im2col()
{
    image im = load_image("data/test/dog.jpg"); 
    matrix col = im2col(im, 3, 2);
    matrix truth_col = load_matrix("data/test/im2col.matrix");
    matrix col2 = im2col(im, 2, 2);
    matrix truth_col2 = load_matrix("data/test/im2col2.matrix");
    TEST(same_matrix(truth_col,   col));
    TEST(same_matrix(truth_col2,  col2));
    FM(col);
    FM(col2);
    FM(truth_col);
    FM(truth_col2);
    FIMG(im);
}

void test_col2im()
{
    image im = load_image("data/test/dog.jpg"); 
    matrix dcol = load_matrix("data/test/dcol.matrix");
    matrix dcol2 = load_matrix("data/test/dcol2.matrix");
    image col2im_res = make_image(im.w, im.h, im.c);
    image col2im_res2 = make_image(im.w, im.h, im.c);
    col2im(dcol, 3, 2, col2im_res);
    col2im(dcol2, 2, 2, col2im_res2);
    matrix col2mat2 = {0};
    col2mat2.rows = col2im_res2.c;
    col2mat2.cols = col2im_res2.w*col2im_res2.h;
    col2mat2.data = col2im_res2.data;

    matrix col2mat = {0};
    col2mat.rows = col2im_res.c;
    col2mat.cols = col2im_res.w*col2im_res.h;
    col2mat.data = col2im_res.data;
    matrix truth_col2mat = load_matrix("data/test/col2mat.matrix");
    matrix truth_col2mat2 = load_matrix("data/test/col2mat2.matrix");
    TEST(same_matrix(truth_col2mat, col2mat));
    TEST(same_matrix(truth_col2mat2, col2mat2));
    FM(dcol);
    FM(col2mat);
    FM(truth_col2mat);
    FM(dcol2);
    FM(col2mat2);
    FM(truth_col2mat2);
    FIMG(im);
}

void test_maxpool_layer_forward()
{
    image im = load_image("data/test/dog.jpg"); 
    matrix im_mat = {0};
    im_mat.rows = 1;
    im_mat.cols = im.w*im.h*im.c;
    im_mat.data = im.data;
    matrix im_mat3 = {0};
    im_mat3.rows = 1;
    im_mat3.cols = im.w*im.h*im.c;
    im_mat3.data = im.data;
    layer max_l = make_maxpool_layer(im.w, im.h, im.c, 2, 2);
    matrix max_out = max_l.forward(max_l, im_mat);
    matrix truth_max_out = load_matrix("data/test/max_out.matrix");
    TEST(same_matrix(truth_max_out, max_out));
    layer max_l3 = make_maxpool_layer(im.w, im.h, im.c, 3, 2);
    matrix max_out3 = max_l3.forward(max_l3, im_mat3);
    matrix truth_max_out3 = load_matrix("data/test/max_out3.matrix");
    TEST(same_matrix(truth_max_out3, max_out3));
    FM(max_out);
    FM(truth_max_out);
    FM(max_out3);
    FM(truth_max_out3);
    FIMG(im);
}

void test_maxpool_layer_backward()
{
    matrix truth_max_out = load_matrix("data/test/max_out.matrix");
    matrix truth_max_out3 = load_matrix("data/test/max_out3.matrix");
    image im = load_image("data/test/dog.jpg"); 
    matrix im_mat = {0};
    im_mat.rows = 1;
    im_mat.cols = im.w*im.h*im.c;
    im_mat.data = im.data;
    layer max_l = make_maxpool_layer(im.w, im.h, im.c, 2, 2);
    max_l.in[0] = im_mat;
    max_l.out[0] = truth_max_out;

    matrix max_delta = load_matrix("data/test/max_delta.matrix");
    matrix prev_max_delta = make_matrix(im_mat.rows, im_mat.cols);

    *max_l.delta = max_delta;
    max_l.backward(max_l, prev_max_delta);
    matrix truth_prev_max_delta = load_matrix("data/test/prev_max_delta.matrix");
    TEST(same_matrix(truth_prev_max_delta, prev_max_delta));

    matrix im_mat3 = {0};
    im_mat3.rows = 1;
    im_mat3.cols = im.w*im.h*im.c;
    im_mat3.data = im.data;
    layer max_l3 = make_maxpool_layer(im.w, im.h, im.c, 3, 2);
    max_l3.in[0] = im_mat3;
    max_l3.out[0] = truth_max_out3;

    matrix max_delta3 = load_matrix("data/test/max_delta3.matrix");
    matrix prev_max_delta3 = make_matrix(im_mat3.rows, im_mat3.cols);

    *max_l3.delta = max_delta3;
    max_l3.backward(max_l3, prev_max_delta3);
    matrix truth_prev_max_delta3 = load_matrix("data/test/prev_max_delta3.matrix");
    TEST(same_matrix(truth_prev_max_delta3, prev_max_delta3));
    FM(max_delta);
    FM(prev_max_delta);
    FM(truth_prev_max_delta);
    FM(max_delta3);
    FM(prev_max_delta3);
    FM(truth_prev_max_delta3);
    FIMG(im);
}

void make_matrix_test()
{
    srand(1);
    matrix a = random_matrix(32, 64, 10);
    matrix b = random_matrix(64, 16, 10);
    matrix at = transpose_matrix(a);
    matrix c = matmul(a, b);
    matrix y = random_matrix(32, 64, 10);
    matrix bias = random_matrix(1, 16, 10);
    matrix dw = random_matrix(64, 16, 10);
    matrix db = random_matrix(1, 16, 10);
    matrix delta = random_matrix(32, 16, 10);
    matrix prev_delta = random_matrix(32, 64, 10);
    matrix y1 = copy_matrix(y);
    axpy_matrix(2, a, y1);
    save_matrix(a, "data/test/a.matrix");
    save_matrix(b, "data/test/b.matrix");
    save_matrix(bias, "data/test/bias.matrix");
    save_matrix(dw, "data/test/dw.matrix");
    save_matrix(db, "data/test/db.matrix");
    save_matrix(at, "data/test/at.matrix");
    save_matrix(delta, "data/test/delta.matrix");
    save_matrix(prev_delta, "data/test/prev_delta.matrix");
    save_matrix(c, "data/test/c.matrix");
    save_matrix(y, "data/test/y.matrix");
    save_matrix(y1, "data/test/y1.matrix");

    matrix alog = copy_matrix(a);
    activate_matrix(alog, LOGISTIC);
    save_matrix(alog, "data/test/alog.matrix");

    matrix arelu = copy_matrix(a);
    activate_matrix(arelu, RELU);
    save_matrix(arelu, "data/test/arelu.matrix");

    matrix alrelu = copy_matrix(a);
    activate_matrix(alrelu, LRELU);
    save_matrix(alrelu, "data/test/alrelu.matrix");

    matrix asoft = copy_matrix(a);
    activate_matrix(asoft, SOFTMAX);
    save_matrix(asoft, "data/test/asoft.matrix");



    matrix glog = copy_matrix(a);
    gradient_matrix(y, LOGISTIC, glog);
    save_matrix(glog, "data/test/glog.matrix");

    matrix grelu = copy_matrix(a);
    gradient_matrix(y, RELU, grelu);
    save_matrix(grelu, "data/test/grelu.matrix");

    matrix glrelu = copy_matrix(a);
    gradient_matrix(y, LRELU, glrelu);
    save_matrix(glrelu, "data/test/glrelu.matrix");

    matrix gsoft = copy_matrix(a);
    gradient_matrix(y, SOFTMAX, gsoft);
    save_matrix(gsoft, "data/test/gsoft.matrix");


    layer l = make_connected_layer(64, 16, LRELU);
    l.w = b;
    l.b = bias;
    l.dw = dw;
    l.db = db;

    matrix out = l.forward(l, a);
    save_matrix(out, "data/test/out.matrix");

    l.delta[0] = delta;
    l.backward(l, prev_delta);
    save_matrix(prev_delta, "data/test/truth_prev_delta.matrix");
    save_matrix(l.dw, "data/test/truth_dw.matrix");
    save_matrix(l.db, "data/test/truth_db.matrix");

    l.update(l, .01, .9, .01);
    save_matrix(l.dw, "data/test/updated_dw.matrix");
    save_matrix(l.db, "data/test/updated_db.matrix");
    save_matrix(l.w, "data/test/updated_w.matrix");
    save_matrix(l.b, "data/test/updated_b.matrix");

    image im = load_image("data/test/dog.jpg"); 
    matrix col = im2col(im, 3, 2);
    matrix col2 = im2col(im, 2, 2);
    save_matrix(col, "data/test/im2col.matrix");
    save_matrix(col2, "data/test/im2col2.matrix");

    matrix dcol = random_matrix(col.rows, col.cols, 10);
    matrix dcol2 = random_matrix(col2.rows, col2.cols, 10);
    image col2im_res = make_image(im.w, im.h, im.c);
    image col2im_res2 = make_image(im.w, im.h, im.c);
    col2im(dcol, 3, 2, col2im_res);
    col2im(dcol2, 2, 2, col2im_res2);
    save_matrix(dcol, "data/test/dcol.matrix");
    save_matrix(dcol2, "data/test/dcol2.matrix");
    matrix col2mat = {0};
    col2mat.rows = col2im_res.c;
    col2mat.cols = col2im_res.w*col2im_res.h;
    col2mat.data = col2im_res.data;
    save_matrix(col2mat, "data/test/col2mat.matrix");
    matrix col2mat2 = {0};
    col2mat2.rows = col2im_res2.c;
    col2mat2.cols = col2im_res2.w*col2im_res2.h;
    col2mat2.data = col2im_res2.data;
    save_matrix(col2mat2, "data/test/col2mat2.matrix");


    // Maxpool Layer Tests

    matrix im_mat = {0};
    im_mat.rows = 1;
    im_mat.cols = im.w*im.h*im.c;
    im_mat.data = im.data;
    layer max_l = make_maxpool_layer(im.w, im.h, im.c, 2, 2);
    matrix max_out = max_l.forward(max_l, im_mat);
    save_matrix(max_out, "data/test/max_out.matrix");

    matrix max_delta = random_matrix(max_out.rows, max_out.cols, 10);
    save_matrix(max_delta, "data/test/max_delta.matrix");

    matrix prev_max_delta = make_matrix(im_mat.rows, im_mat.cols);
    *max_l.delta = max_delta;
    max_l.backward(max_l, prev_max_delta);
    save_matrix(prev_max_delta, "data/test/prev_max_delta.matrix");

    matrix im_mat3 = {0};
    im_mat3.rows = 1;
    im_mat3.cols = im.w*im.h*im.c;
    im_mat3.data = im.data;
    layer max_l3 = make_maxpool_layer(im.w, im.h, im.c, 3, 2);
    matrix max_out3 = max_l.forward(max_l3, im_mat3);
    save_matrix(max_out3, "data/test/max_out3.matrix");

    matrix max_delta3 = random_matrix(max_out3.rows, max_out3.cols, 10);
    save_matrix(max_delta3, "data/test/max_delta3.matrix");

    matrix prev_max_delta3 = make_matrix(im_mat3.rows, im_mat3.cols);
    *max_l3.delta = max_delta3;
    max_l.backward(max_l3, prev_max_delta3);
    save_matrix(prev_max_delta3, "data/test/prev_max_delta3.matrix");


    // Batchnorm Tests

    matrix mu_a   = mean(a, 1);
    matrix mu_a_s = mean(a, 8);

    matrix sig_a   =  variance(a, mu_a, 1);
    matrix sig_a_s =  variance(a, mu_a_s, 8);

    matrix norm_a = normalize(a, mu_a, sig_a, 1);
    matrix norm_a_s = normalize(a, mu_a_s, sig_a_s, 8);

    save_matrix(mu_a, "data/test/mu_a.matrix");
    save_matrix(mu_a_s, "data/test/mu_a_s.matrix");
    save_matrix(sig_a, "data/test/sig_a.matrix");
    save_matrix(sig_a_s, "data/test/sig_a_s.matrix");
    save_matrix(norm_a, "data/test/norm_a.matrix");
    save_matrix(norm_a_s, "data/test/norm_a_s.matrix");

    matrix dm = delta_mean(y, sig_a, 1);
    matrix dm_s = delta_mean(y, sig_a_s, 8);

    save_matrix(dm, "data/test/dm.matrix");
    save_matrix(dm_s, "data/test/dm_s.matrix");

    matrix dv = delta_variance(y, a, mu_a, sig_a, 1);
    matrix dv_s = delta_variance(y, a, mu_a_s, sig_a_s, 8);

    save_matrix(dv, "data/test/dv.matrix");
    save_matrix(dv_s, "data/test/dv_s.matrix");

    matrix dbn = delta_batch_norm(y, dm, dv, mu_a, sig_a, a, 1);
    matrix dbn_s = delta_batch_norm(y, dm_s, dv_s, mu_a_s, sig_a_s, a, 8);
    save_matrix(dbn, "data/test/dbn.matrix");
    save_matrix(dbn_s, "data/test/dbn_s.matrix");
}

void test_matrix_speed()
{
    int i;
    int n = 128;
    matrix a = random_matrix(512, 512, 1);
    matrix b = random_matrix(512, 512, 1);
    double start = what_time_is_it_now();
    for(i = 0; i < n; ++i){
        matrix d = matmul(a,b);
        FM(d);
    }
    printf("Matmul elapsed %lf sec\n", what_time_is_it_now() - start);
    start = what_time_is_it_now();
    for(i = 0; i < n; ++i){
        matrix at = transpose_matrix(a);
        FM(at);
    }
    printf("Transpose elapsed %lf sec\n", what_time_is_it_now() - start);
}

void run_tests()
{
    //make_matrix_test();
    test_copy_matrix();
    test_axpy_matrix();
    test_transpose_matrix();
    test_matmul();
    test_activate_matrix();
    test_gradient_matrix();
    test_connected_layer();
    test_im2col();
    test_col2im();
    test_maxpool_layer_forward();
    test_maxpool_layer_backward();
    test_matrix_mean();
    test_matrix_variance();
    test_matrix_normalize();
    test_matrix_delta_mean();
    test_matrix_delta_variance();
    test_matrix_delta_normalize();
    test_matrix_speed();
    printf("%d tests, %d passed, %d failed\n", tests_total, tests_total-tests_fail, tests_fail);
}

//imghw

void test_hw0()
{
    test_get_pixel();
    test_set_pixel();
    test_copy();
    test_shift();
    test_clamp();
    test_grayscale();
    test_rgb_to_hsv();
    test_hsv_to_rgb();
    printf("%d tests, %d passed, %d failed\n", tests_total, tests_total-tests_fail, tests_fail);
}
void test_hw1()
{
    test_nn_interpolate();
    test_nn_resize();
    test_bl_interpolate();
    test_bl_resize();
    test_multiple_resize();
    printf("%d tests, %d passed, %d failed\n", tests_total, tests_total-tests_fail, tests_fail);
}
void test_hw2()
{
    test_gaussian_filter();
    test_sharpen_filter();
    test_emboss_filter();
    test_highpass_filter();
    test_convolution();
    test_gaussian_blur();
    test_hybrid_image();
    test_frequency_image();
    test_sobel();
    printf("%d tests, %d passed, %d failed\n", tests_total, tests_total-tests_fail, tests_fail);
}

void test_matchain(){
	int i,n=5;
	matrix *chain=calloc(n,sizeof(matrix));
	int r,c;
	r=3+(rand()&3);
	for(i=0;i<n;++i){
		c=3+(rand()&3);
		chain[i]=RM(r,c,f64rand());
		r=c;
	}
	
	
	matrix res1,tmp;
	res1=matmul(chain[0],chain[1]);
	for(i=2;i<n;++i){
		tmp=res1;
		res1=matmul(tmp,chain[i]);
		FM(tmp);
	}
	printf("%s:\n","res1");PM(res1);
	FM(res1);
	matrix res=matchain(chain,n);
	printf("%s:\n","res");PM(res);
	FM(res);
	free(chain);
}


void test_hw3()
{
    test_structure();
    test_cornerness();
    test_projection();
    test_compute_homography();
	test_matchain();
    printf("%d tests, %d passed, %d failed\n", tests_total, tests_total-tests_fail, tests_fail);
}
void make_hw4_tests()
{
    image dots = load_image("data/dots.png");
    image intdot = make_integral_image(dots,0);
    save_image_binary(intdot, "data/dotsintegral.bin");

    image dogbw = load_image("data/dogbw.png");
    image intdog = make_integral_image(dogbw,0);
    save_image_binary(intdog, "data/dogintegral.bin");

    image dog = load_image("data/dog.jpg");
    image smooth = box_filter_image(dog, 15);
    save_png(smooth, "data/dogbox");

    image smooth_c = center_crop(smooth);
    save_png(smooth_c, "data/dogboxcenter");

    image doga = load_image("data/dog_a_small.jpg");
    image dogb = load_image("data/dog_b_small.jpg");
    image structure = time_structure_matrix(dogb, doga, 15);
    save_image_binary(structure, "data/structure.bin");

    image velocity = velocity_image(structure, 5);
    save_image_binary(velocity, "data/velocity.bin");
}
void test_integral_image()
{
    image dots = load_image("data/dots.png");
    image intdot = make_integral_image(dots,0);
    image intdot_t = load_image_binary("data/dotsintegral.bin");
    TEST(same_image(intdot, intdot_t, EPS));

    image dog = load_image("data/dogbw.png");
    image intdog = make_integral_image(dog,0);
    image intdog_t = load_image_binary("data/dogintegral.bin");
    TEST(same_image(intdog, intdog_t, .6));
}
void test_exact_box_filter_image()
{
    image dog = load_image("data/dog.jpg");
    image smooth = box_filter_image(dog, 15);
	save_png(smooth,"dogbox");
    image smooth_t = load_image("data/dogbox.png");
    //printf("avg origin difference test: %f\n", avg_diff(smooth, dog));
    //printf("avg smooth difference test: %f\n", avg_diff(smooth, smooth_t));
    TEST(same_image(smooth, smooth_t, EPS*2));
}

void test_good_enough_box_filter_image()
{
    image dog = load_image("data/dog.jpg");
    image smooth = box_filter_image(dog, 15);
    image smooth_c = center_crop(smooth);
    image smooth_t = load_image("data/dogboxcenter.png");
    printf("avg origin difference test: %f\n", avg_diff(smooth_c, center_crop(dog)));
    printf("avg smooth difference test: %f\n", avg_diff(smooth_c, smooth_t));
    TEST(same_image(smooth_c, smooth_t, EPS*2));
}
void test_structure_image()
{
    image doga = load_image("data/dog_a_small.jpg");
    image dogb = load_image("data/dog_b_small.jpg");
    image structure = time_structure_matrix(dogb, doga, 15);
    image structure_t = load_image_binary("data/structure.bin");
    TEST(same_image(center_crop(structure), center_crop(structure_t), EPS));
}
void test_velocity_image()
{
    image structure = load_image_binary("data/structure.bin");
    image velocity = velocity_image(structure, 5);
    image velocity_t = load_image_binary("data/velocity.bin");
    TEST(same_image(velocity, velocity_t, EPS));
}
void test_hw4()
{
	//~ make_hw4_tests();
    test_integral_image();
    test_exact_box_filter_image();
    test_good_enough_box_filter_image();
    test_structure_image();
    test_velocity_image();
    printf("%d tests, %d passed, %d failed\n", tests_total, tests_total-tests_fail, tests_fail);
}
