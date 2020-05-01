# Homework 0 #

Welcome friends,

For the first assignment we'll just get to know the codebase a little bit and practice manipulating images, transforming things, breaking stuff, it should be fun!

## Image basics ##

We have a pretty basic datastructure to store images in our library. The `image` struct stores the image metadata like width, height, and number of channels. It also contains the image data stored as a floating point array. You can check it out in `src/image.h`, it looks like this:

    typedef struct{
        int h,w,c;
        float *data;
    } image;

We have also provided some functions for loading and saving images. Use the function:

    image im = load_image("image.jpg");

to load a new image. To save an image use:

    save_image(im, "output");

which will save the image as `output.jpg`. If you want to make a new image with dimensions Width x Height x Channels you can call:

    image im = make_image(w,h,c);

You should also use: 

    free_image(im);

when you are done with an image. So it goes away. You can check out how all this is implemented in `src/load_image.c`. You probably shouldn't change anything in this file. We use the `stb_image` libary for the actual loading and saving of jpgs because that is, like, REALLY complicated. I think. I've never tried. Anywho....

You'll be modifying the file `src/hw0/process_image.c`. We've also included a python compatability library. `uwimg.py` includes the code to access your C library from python. `tryhw0.py` has some example code you can run. We will build the library using `make`. Simply run the command:

    make
    
after you make any changes to the code. Then you can quickly test your changes by running:

    ./uwimg test

You can also try running the example python code to generate some images:

    python tryhw0.py

## 0.0 Getting and setting pixels ##

The most basic operation we want to do is change the pixels in an image. As we talked about in class, we represent an image as a 3 dimensional tensor. We have spatial information as well as multiple channels which combine together to form a color image:

![RGB format](./figs/rgb.png)

The convention is that the coordinate system starts at the top left of the image, like so:

![Image coordinate system](./figs/coords.png)

In our `data` array we store the image in `CHW` format. The first pixel in data is at channel 0, row 0, column 0. The next pixel is channel 0, row 0, column 1, then channel 0, row 0, column 2, etc.

Your first task is to fill out these two functions in `src/hw0/process_image.c`:

    float get_pixel(image im, int x, int y, int c);
    void set_pixel(image im, int x, int y, int c, float v);

`get_pixel` should return the pixel value at column `x`, row `y`, and channel `c`. `set_pixel` should set the pixel to the value `v`. You will need to do bounds checking to make sure the coordinates are valid for the image. `set_pixel` should simply return without doing anything if you pass in invalid coordinates. For `get_pixel` we will perform padding to the image. There are a number of possible padding strategies:

![Image padding strategies](./figs/pad.png)

We will use the `clamp` padding strategy. This means that if the programmer asks for a pixel at column -3, use column 0, or if they ask for column 300 and the image is only 256x256 you will use column 255 (because of zero-based indexing).

We can test out our pixel-setting code on the dog image by removing all of the red channel. See line 3-8 in `tryhw0.py`:

    # 1. Getting and setting pixels
    im = load_image("data/dog.jpg")
    for row in range(im.h):
        for col in range(im.w):
            set_pixel(im, row, col, 0, 0)
    save_image(im, "figs/dog_no_red")

Then try running it. Check out our very not red dog:

![](./figs/dog_no_red.jpg)


## 0.1 Copying images ##

Sometimes you have an image and you want to copy it! To do this we should make a new image of the same size and then fill in the data array in the new image. You could do this by getting and setting pixels, by looping over the whole array and just copying the floats (pop quiz: if the image is 256x256x3, how many total pixels are there?), or by using the built-in memory copying function `memcpy`.

Fill in the function `image copy_image(image im)` in `src/hw0/process_image.c` with your code.

## 0.2 Grayscale image ##

Now let's start messing with some images! People like making images grayscale. It makes them look... old? Or something? Let's do it.

Remember how humans don't see all colors equally? Here's the chart to remind you:

![Eye sensitivity to different wavelengths](./figs/sensitivity.png)

This actually makes a huge difference in practice. Here's a colorbar we may want to convert:

![Color bar](./figs/colorbar.png)

If we convert it using an equally weighted mean K = (R+G+B)/3 we get a conversion that doesn't match our perceptions of the given colors:

![Averaging grayscale](./figs/avggray.jpg)

Instead we are going to use a weighted sum. Now, there are a few ways to do this. If we wanted the most accurate conversion it would take a fair amount of work. sRGB uses [gamma compression][1] so we would first want to convert the color to linear RGB and then calculate [relative luminance](https://en.wikipedia.org/wiki/Relative_luminance).

But we don't care about being toooo accurate so we'll just do the quick and easy version instead. Video engineers use a calculation called [luma][2] to find an approximation of perceptual intensity when encoding video signal, we'll use that to convert our image to grayscale. It operates directly on the gamma compressed sRGB values that we already have! We simply perform a weighted sum:

    Y' = 0.299 R' + 0.587 G' + .114 B'

Using this conversion technique we get a pretty good grayscale image! Now we can run `tryhw0.py` to output `graybar.jpg`. See lines 10-13:

    # 3. Grayscale image
    im = load_image("data/colorbar.png")
    graybar = rgb_to_grayscale(im)
    save_image(graybar, "graybar")

![Grayscale colorbars](./figs/gray.png)

Implement this conversion for the function `rgb_to_grayscale`. Return a new image that is the same size but only one channel containing the calculated luma values.

## 0.3 Shifting the image colors ##

Now let's write a function to add a constant factor to a channel in an image. We can use this across every channel in the image to make the image brighter or darker. We could also use it to, say, shift an image to be more or less of a given color.

Fill in the code for `void shift_image(image im, int c, float v);`. It should add `v` to every pixel in channel `c` in the image. Now we can try shifting all the channels in an image by `.4` or 40%. See lines 15-20 in `tryhw0.py`:

    # 4. Shift Image
    im = load_image("data/dog.jpg")
    shift_image(im, 0, .4)
    shift_image(im, 1, .4)
    shift_image(im, 2, .4)
    save_image(im, "overflow")

But wait, when we look at the resulting image `overflow.jpg` we see something bad has happened! The light areas of the image went past 1 and when we saved the image back to disk it overflowed and made weird patterns:

![Overflow](./figs/overflow.jpg)

## 0.4 Clamping the image values

Our image pixel values have to be bounded. Generally images are stored as byte arrays where each red, green, or blue value is an unsigned byte between 0 and 255. 0 represents none of that color light and 255 represents that primary color light turned up as much as possible.

We represent our images using floating point values between 0 and 1. However, we still have to convert between our floating point representation and the byte arrays that are stored on disk. In the example above, our pixel values got above 1 so when we converted them back to byte arrays and saved them to disk they overflowed the byte data type and went back to very small values. That's why the very bright areas of the image looped around and became dark.

We want to make sure the pixel values in the image stay between 0 and 1. Implement clamping on the image so that any value below zero gets set to zero and any value above 1 gets set to one. Fill in `void clamp_image(image im);` to modify the image in-place. Then when we clamp the shifted image and save it we see much better results, see lines 22-24 in `tryhw0.py`:

    # 5. Clamp Image
    clamp_image(im)
    save_image(im, "fixed")

and the resulting image, `fixed.jpg`:

![](./figs/fixed.jpg)

## 0.5 RGB to Hue, Saturation, Value ##

So far we've been focussing on RGB and grayscale images. But there are other colorspaces out there too we may want to play around with. Like [Hue, Saturation, and Value (HSV)](https://en.wikipedia.org/wiki/HSL_and_HSV). We will be translating the cubical colorspace of sRGB to the cylinder of hue, saturation, and value:

![RGB HSV conversion](./figs/convert.png)

[Hue](https://en.wikipedia.org/wiki/Hue) can be thought of as the base color of a pixel. [Saturation](https://en.wikipedia.org/wiki/Colorfulness#Saturation) is the intensity of the color compared to white (the least saturated color). The [Value](https://en.wikipedia.org/wiki/Lightness) is the perception of brightness of a pixel compared to black. You can try out this [demo](http://math.hws.edu/graphicsbook/demos/c2/rgb-hsv.html) to get a better feel for the differences between these two colorspaces. For a geometric interpretation of what this transformation:

![RGB to HSV geometry](./figs/rgbtohsv.png)

Now, to be sure, there are [lots of issues](http://poynton.ca/notes/colour_and_gamma/ColorFAQ.html#RTFToC36) with this colorspace. But it's still fun to play around with and relatively easy to implement. The easiest component to calculate is the Value, it's just the largest of the 3 RGB components:

    V = max(R,G,B)

Next we can calculate Saturation. This is a measure of how much color is in the pixel compared to neutral white/gray. Neutral colors have the same amount of each three color components, so to calculate saturation we see how far the color is from being even across each component. First we find the minimum value

    m = min(R,G,B)

Then we see how far apart the min and max are:

    C = V - m

and the Saturation will be the ratio between the difference and how large the max is:

    S = C / V

Except if R, G, and B are all 0. Because then V would be 0 and we don't want to divide by that, so just set the saturation 0 if that's the case.

Finally, to calculate Hue we want to calculate how far around the color hexagon our target color is.

![color hex](./figs/hex.png)

We start counting at Red. Each step to a point on the hexagon counts as 1 unit distance. The distance between points is given by the relative ratios of the secondary colors. We can use the following formula from [Wikipedia](https://en.wikipedia.org/wiki/HSL_and_HSV#Hue_and_chroma):

<img src="./figs/eq.svg" width="256">

There is no "correct" Hue if C = 0 because all of the channels are equal so the color is a shade of gray, right in the center of the cylinder. However, for now let's just set H = 0 if C = 0 because then your implementation will match mine.

Notice that we are going to have H = \[0,1) and it should circle around if it gets too large or goes negative. Thus we check to see if it is negative and add one if it is. This is slightly different than other methods where H is between 0 and 6 or 0 and 360. We will store the H, S, and V components in the same image, so simply replace the R channel with H, the G channel with S, etc.

## 0.6 HSV to RGB ##

Ok, now do it all backwards in `hsv_to_rgb`!

Finally, when your done we can mess with some images! In `tryhw0.py` we convert an image to HSV, increase the saturation, then convert it back, lines 26-32:

    # 6-7. Colorspace and saturation
    im = load_image("data/dog.jpg")
    rgb_to_hsv(im)
    shift_image(im, 1, .2)
    clamp_image(im)
    hsv_to_rgb(im)
    save_image(im, "dog_saturated")

![Saturated dog picture](./figs/dog_saturated.jpg)

Hey that's exciting! Play around with it a little bit, see what you can make. Note that with the above method we do get some artifacts because we are trying to increase the saturation in areas that have very little color. Instead of shifting the saturation, you could scale the saturation by some value to get smoother results!

## 0.7 A small amount of extra credit ##

Implement `void scale_image(image im, int c, float v);` to scale a channel by a certain amount. This will give us better saturation results. Note, you will have to add the necessary lines to the header and python library, it should be very similar to what's already there for `shift_image`. Now if we scale saturation by `2` instead of just shifting it all up we get much better results:

    im = load_image("data/dog.jpg")
    rgb_to_hsv(im)
    scale_image(im, 1, 2)
    clamp_image(im)
    hsv_to_rgb(im)
    save_image(im, "dog_scale_saturated")
    
![Dog saturated smoother](./figs/dog_scale_saturated.jpg)

## 0.8 Super duper extra credit ##

Implement RGB to [Hue, Chroma, Lightness](https://en.wikipedia.org/wiki/CIELUV#Cylindrical_representation_.28CIELCH.29), a perceptually more accurate version of Hue, Saturation, Value. Note, this will involve gamma decompression, converting to CIEXYZ, converting to CIELUV, converting to HCL, and the reverse transformations. The upside is a similar colorspace to HSV but with better perceptual properties!

## Turn it in ##

You only need to turn in one file, your `process_image.c`. Use the dropbox link on the class website.

[1]: https://en.wikipedia.org/wiki/SRGB#The_sRGB_transfer_function_("gamma")
[2]: https://en.wikipedia.org/wiki/Luma_(video)



# Homework 1 #

Welcome friends,

It's time for assignment 1! This one may be a little harder than the last one so remember to start early and start often! In order to make grading easier, please only edit the files we mention. You will submit the `resize_image.c` file on Canvas.

To test your homework, make sure you have run `make` to recompile, then run `./uwimg test hw1`

## 1. Image resizing ##

We've been talking a lot about resizing and interpolation in class, now's your time to do it! To resize we'll need some interpolation methods and a function to create a new image and fill it in with our interpolation methods.

### 1.1 Nearest Neighbor Interpolation ###

Fill in `float nn_interpolate(image im, float x, float y, int c);` in `src/resize_image.c`

It should perform nearest neighbor interpolation. Remember to use the closest `int`, not just type-cast because in C that will truncate towards zero.

### 1.2 Nearest Neighbor Resizing ###

Fill in `image nn_resize(image im, int w, int h);`. It should:
- Create a new image that is `w x h` and the same number of channels as `im`
- Loop over the pixels and map back to the old coordinates
- Use nearest-neighbor interpolate to fill in the image

Now you should be able to run the following `python` command:

    from uwimg import *
    im = load_image("data/dogsmall.jpg")
    a = nn_resize(im, im.w*4, im.h*4)
    save_image(a, "dog4x-nn")

Your image should look something like:

![blocky dog](./figs/dog4x-nn.png)

### 1.3 Bilinear Interpolation ###

Finally, fill in the similar functions `bilinear_interpolate` and....

### 1.4 Bilinear Resizing ###

`bilinear_resize` to perform bilinear interpolation. Try it out again in `python`:

    from uwimg import *
    im = load_image("data/dogsmall.jpg")
    a = bilinear_resize(im, im.w*4, im.h*4)
    save_image(a, "dog4x-bl")

![smooth dog](./figs/dog4x-bl.png)

These functions will work fine for small changes in size, but when we try to make our image smaller, say a thumbnail, we get very noisy results:

    from uwimg import *
    im = load_image("data/dog.jpg")
    a = nn_resize(im, im.w//7, im.h//7)
    save_image(a, "dog7th-bl")

![jagged dog thumbnail](./figs/dog7th-nn.png)

As we discussed, we need to filter before we do this extreme resize operation!


## Turn it in ##

Turn in your `resize_image.c` on canvas under Assignment 1.

# Homework 2 #

If we want to perform the large shrinking operations that we talked about in the last homework we need to first smooth our image. We'll start out by filtering the image with a box filter. There are very fast ways of performing this operation but instead, we'll do the naive thing and implement it as a convolution because it will generalize to other filters as well!

To start, you may need to `git pull` the latest changes from GitHub to make sure your code is up to date. Then make sure to `make` (or even better, `make clean` and then `make`) so that your code is recompiled. You can run `./uwimg test hw2` to test your progress as you work.
    
### 2.1 Create your box filter ###

Ok, bear with me. We want to create a box filter, which as discussed in class looks like this:

![box filter](./figs/boxfilter.png)

One way to do this is make an image, fill it in with all 1s, and then normalize it. That's what we'll do because the normalization function may be useful in the future!

First fill in `void l1_normalize(image im)`. This should normalize an image to sum to 1.

Next fill in `image make_box_filter(int w)`. We will only use square box filters so just make your filter `w x w`. It should be a square image with one channel with uniform entries that sum to 1.

### 2.2 Write a convolution function ###

Now it's time to fill in `image convolve_image(image im, image filter, int preserve)`. For this function we have a few scenarios. With normal convolutions we do a weighted sum over an area of the image. With multiple channels in the input image there are a few possible cases we want to handle:

- If `filter` and `im` have the same number of channels then it's just a normal convolution. We sum over spatial and channel dimensions and produce a 1 channel image. UNLESS:
- If `preserve` is set to 1 we should produce an image with the same number of channels as the input. This is useful if, for example, we want to run a box filter over an RGB image and get out an RGB image. This means each channel in the image will be filtered by the corresponding channel in the filter. UNLESS:
- If the `filter` only has one channel but `im` has multiple channels we want to apply the filter to each of those channels. Then we either sum between channels or not depending on if `preserve` is set.

Also, `filter` better have either the same number of channels as `im` or have 1 channel. I check this with an `assert`.

We are calling this a convolution but you don't need to flip the filter or anything (we're actually doing a cross-correlation). Just apply it to the image as we discussed in class:

![covolution](./figs/convolution.png)

Once you are done, test out your convolution by filtering our image! We need to use `preserve` because we want to produce an image that is still RGB.

    from uwimg import *
    im = load_image("data/dog.jpg")
    f = make_box_filter(7)
    blur = convolve_image(im, f, 1)
    save_image(blur, "dog-box7")

We'll get some output that looks like this:

![covolution](./figs/dog-box7.png)

Now we can use this to perform our thumbnail operation:

    from uwimg import *
    im = load_image("data/dog.jpg")
    f = make_box_filter(7)
    blur = convolve_image(im, f, 1)
    thumb = nn_resize(blur, blur.w//7, blur.h//7)
    save_image(thumb, "dogthumb")

![covolution](./figs/dogthumb.png)

Look at how much better our new resized thumbnail is!

Resize                     |  Blur and Resize
:-------------------------:|:-------------------------:
![](./figs/dog7th-nn.png)    | ![](./figs/dogthumb.png)

### 2.2 Make some more filters and try them out! ###

Fill in the functions `image make_highpass_filter()`, `image make_sharpen_filter()`, and `image make_emboss_filter()` to return the example kernels we covered in class. Try them out on some images! After you have, answer Question 2.2.1 and 2.2.2 in the source file (put your answer just right there)

Highpass                   |  Sharpen                  | Emboss
:-------------------------:|:-------------------------:|:--------------------|
![](./figs/highpass.png)     | ![](./figs/sharpen.png)     | ![](./figs/emboss.png)

### 2.3 Implement a Gaussian kernel ###

Implement `image make_gaussian_filter(float sigma)` which will take a standard deviation value and return a filter that smooths using a gaussian with that sigma. How big should the filter be, you ask? 99% of the probability mass for a gaussian is within +/- 3 standard deviations so make the kernel be 6 times the size of sigma. But also we want an odd number, so make it be the next highest odd integer from 6x sigma.

We need to fill in our kernel with some values. Use the probability density function for a 2d gaussian:

![2d gaussian](./figs/2dgauss.png)

Technically this isn't perfect, what we would really want to do is integrate over the area covered by each cell in the filter. But that's much more complicated and this is a decent estimate. Remember though, this is a blurring filter so we want all the weights to sum to 1. If only we had a function for that....

Now you should be able to try out your new blurring function! It should have much less noise than the box filter:

    from uwimg import *
    im = load_image("data/dog.jpg")
    f = make_gaussian_filter(2)
    blur = convolve_image(im, f, 1)
    save_image(blur, "dog-gauss2")

![blurred dog](./figs/dog-gauss2.png)

## 2.4 Hybrid images ##

Gaussian filters are cool because they are a true low-pass filter for the image. This means when we run them on an image we only get the low-frequency changes in an image like color. Conversely, we can subtract this low-frequency information from the original image to get the high frequency information!

Using this frequency separation we can do some pretty neat stuff. For example, check out [this tutorial on retouching skin](https://petapixel.com/2015/07/08/primer-using-frequency-separation-in-photoshop-for-skin-retouching/) in Photoshop (but only if you want to).

We can also make [really trippy images](http://cvcl.mit.edu/hybrid/OlivaTorralb_Hybrid_Siggraph06.pdf) that look different depending on if you are close or far away from them. That's what we'll be doing. They are hybrid images that take low frequency information from one image and high frequency info from another. Here's a picture of.... what exactly?

Small                     |  Medium | Large
:-------------------------:|:-------:|:------------------:
![](./figs/marilyn-einstein-small.png)   | ![](./figs/marilyn-einstein-medium.png) | ![](./figs/marilyn-einstein.png)

If you don't believe my resizing check out `figs/marilyn-einstein.png` and view it from far away and up close. Sorta neat, right?

Your job is to produce a similar image. But instead of famous dead people we'll be using famous fictional people! In particular, we'll be exposing the secret (but totally canon) sub-plot of the Harry Potter franchise that Dumbledore is a time-traveling Ron Weasely. Don't trust me?? The images don't lie! Wake up sheeple!


Small                     | Large
:-------------------------:|:------------------:
![](./figs/ronbledore-small.jpg)   | ![](./figs/ronbledore.jpg) 

For this task you'll have to extract the high frequency and low frequency from some images. You already know how to get low frequency, using your gaussian filter. To get high frequency you just subtract the low frequency data from the original image.

Fill in `image add_image(image a, image b)` and `image sub_image(image a, image b)` so we can perform our transformations. They should probably include some checks that the images are the same size and such. Now we should be able to run something like this:

    from uwimg import *
    im = load_image("data/dog.jpg")
    f = make_gaussian_filter(2)
    lfreq = convolve_image(im, f, 1)
    hfreq = im - lfreq
    reconstruct = lfreq + hfreq
    save_image(lfreq, "low-frequency")
    save_image(hfreq, "high-frequency")
    save_image(reconstruct, "reconstruct")


Low frequency           |  High frequency | Reconstruction
:-------------------------:|:-------:|:------------------:
![](./figs/low-frequency.png)   | ![](./figs/high-frequency.png) | ![](./figs/reconstruct.png)

Note, the high-frequency image overflows when we save it to disk? Is this a problem for us? Why or why not?

Use these functions to recreate your own Ronbledore image. You will need to tune your standard deviations for the gaussians you use. You will probably need different values for each image to get it to look good.

## 2.5 Sobel filters ##

The [Sobel filter](https://www.researchgate.net/publication/239398674_An_Isotropic_3x3_Image_Gradient_Operator) is cool because we can estimate the gradients and direction of those gradients in an image. They should be straightforward now that you all are such pros at image filtering.

### 2.5.1 Make the filters ###

First implement the functions to make our sobel filters. They are for estimating the gradient in the x and y direction:

Gx                 |  Gy 
:-----------------:|:------------------:
![](./figs/gx.png)   |  ![](./figs/gy.png)


### 2.5.2 One more normalization... ###

To visualize our sobel operator we'll want another normalization strategy, [feature normalization](https://en.wikipedia.org/wiki/Feature_scaling). This strategy is simple, we just want to scale the image so all values lie between [0-1]. In particular we will be [rescaling](https://en.wikipedia.org/wiki/Feature_scaling#Rescaling) the image by subtracting the minimum from all values and dividing by the range of the data. If the range is zero you should just set the whole image to 0 (don't divide by 0 that's bad).

### 2.5.3 Calculate gradient magnitude and direction ###

Fill in the function `image *sobel_image(image im)`. It should return two images, the gradient magnitude and direction. The strategy can be found [here](https://en.wikipedia.org/wiki/Sobel_operator#Formulation). We can visualize our magnitude using our normalization function:

    from uwimg import *
    im = load_image("data/dog.jpg")
    res = sobel_image(im)
    mag = res[0]
    feature_normalize(mag)
    save_image(mag, "magnitude")

Which results in:

![](./figs/magnitude.png)

### 2.5.4 Make a colorized representation ###

Now using your sobel filter try to make a cool, stylized one. Fill in the function `image colorize_sobel(image im)`. I used the magnitude to specify the saturation and value of an image and the angle to specify the hue but you can do whatever you want (as long as it looks cool). I also used some smoothing:

![](./figs/lcolorized.png)

## Turn it in ##

Turn in your `filter_image.c` on canvas under Assignment 2.


![panorama of field](./figs/field_panorama.jpg)

# Homework 3 #

Welcome friends,

It's time for assignment 3!

## Let's make a panorama! ##

This homework covers a lot, including finding keypoints in an image, describing those key points, matching them to those points in another image, computing the transform from one image to the other, and stitching them together into a panorama.

The high-level algorithm is already done for you! You can find it near the bottom of `src/hw3/panorama_image.c`, it looks approximately like:


    image panorama_image(image a, image b, float sigma, float thresh, int nms, float inlier_thresh, int iters, int cutoff)
    {
        // Calculate corners and descriptors
        descriptor *ad = harris_corner_detector(a, sigma, thresh, nms, &an);
        descriptor *bd = harris_corner_detector(b, sigma, thresh, nms, &bn);

        // Find matches
        match *m = match_descriptors(ad, an, bd, bn, &mn);

        // Run RANSAC to find the homography
        matrix H = RANSAC(m, mn, inlier_thresh, iters, cutoff);

        // Stitch the images together with the homography
        image comb = combine_images(a, b, H);
        return comb;
    }

So we'll find the corner points in an image using a Harris corner detector. Then we'll match together the descriptors of those corners. We'll use RANSAC to estimate a projection from one image coordinate system to the other. Finally, we'll stitch together the images using this projection.

First we need to find those corners!

## 3. Harris corner detection ##

We'll be implementing Harris corner detection as discussed in class. The basic algorithm is:

    Calculate image derivatives Ix and Iy.
    Calculate measures IxIx, IyIy, and IxIy.
    Calculate structure matrix components as weighted sum of nearby measures.
    Calculate Harris "cornerness" as estimate of 2nd eigenvalue: det(S) - α trace(S)^2, α = .06
    Run non-max suppression on response map

## 3.1 Compute the structure matrix ##

Fill in `image structure_matrix(image im, float sigma);` in `harris_image.c`. This will perform the first 3 steps of the algorithm: calculating derivatives, the corresponding measures, and the weighted sum of nearby derivative information. As discussed in class, this weighted sum can be easily computed with a Gaussian blur.

### 3.1b Optional: Make a fast smoother ###

If you want a fast corner detector you can decompose the Gaussian blur from one large 2d convolution to 2 1d convolutions. Instead of using an N x N filter you can convolve with a 1 x N filter followed by the same filter flipped to be N x 1.

Fill in `image make_1d_gaussian(float sigma)` and `image smooth_image(image im, float sigma)` to use this decomposed Gaussian smoothing.

## 3.2 Computer cornerness from structure matrix ##

Fill in `image cornerness_response(image S)`.

## 3.3 Non-maximum suppression ##

We only want local maximum responses to our corner detector so that the matching is easier. Fill in `image nms_image(image im, int w)`.

For every pixel in `im`, check every neighbor within `w` pixels (Chebyshev distance). Equivalently, check the `2w+1` window centered at each pixel. If any responses are stronger, suppress that pixel's response (set it to a very low negative number).

## 3.4 Complete the Harris detector ##

Fill in the missing sections of `descriptor *harris_corner_detector(image im, float sigma, float thresh, int nms, int *n)`. The function should return an array of descriptors for corners in the image. Code for calculating the descriptors is provided. Also, set the integer `*n` to be the number of corners found in the image.

After you complete this function you should be able to calculate corners and descriptors for an image! Try running:

    im = load_image("data/Rainier1.png")
    detect_and_draw_corners(im, 2, 50, 3)
    save_image(im, "corners")

This will detect corners using a Gaussian window of 2 sigma, a "cornerness" threshold of 100, and an nms distance of 3 (or window of 7x7). It should give you something like this:

![rainier corners](./figs/corners.jpg)

Corners are marked with the crosses. They seem pretty sensible! Lots of corners near where snow meets rock and such. Try playing with the different values to see how the affect our corner detector.

## Patch matching ##

To get a panorama we have to match up the corner detections with their appropriate counterpart in the other image. The descriptor code is already written for you. It consists of nearby pixels except with the center pixel value subtracted. This gives us some small amount of invariance to lighting conditions.

The rest of the homework takes place in `src/panorama_image.c`.

## 3.5 Distance metric ##
For comparing patches we'll use L1 distance. Squared error (L2 distance) can be problematic with outliers as we saw in class. We don't want a few rogue pixels to throw off our matching function. L1 distance (sum absolute difference) is better behaved with some outliers.

Implement float `l1_distance(float *a, float *b, int n)` between two vectors of floats. The vectors and how many values they contain is passed in.

## 3.6 Matching descriptors ##

### 3.6.1 Find the best matches from a to b ###

First we'll look through descriptors for `image a` and find their best match with descriptors from `image b`. Fill in the first `TODO` in `match *match_descriptors(descriptor *a, int an, descriptor *b, int bn, int *mn)`.

## 3.6.2 Eliminate multiple matches to the same descriptor in b  ##

Each descriptor in `a` will only appear in one match. But several of them may match with the same descriptor in `b`. This can be problematic. Namely, if a bunch of matches go to the same point there is an easy homography to estimate that just shrinks the whole image down to one point to project from `a` to `b`. But we know that's wrong. So let's just get rid of these duplicate matches and make our matches be one-to-one.

To do this, sort the matches based on distance so shortest distance is first. I've already implemented a comparator for you to use, just look up how to properly apply `qsort` if you don't know already. Next, loop through the matches in order and keep track of which elements in `b` we've seen. If we see one for a second (or third, etc.) time, throw it out! To throw it out, just shift the other elements forward in the list and then remember how many one-to-one matches we have at the end. This can be done with a single pass through the data.

Once this is done we can show the matches we discover between the images:

    a = load_image("data/Rainier1.png")
    b = load_image("data/Rainier2.png")
    m = find_and_draw_matches(a, b, 2, 50, 3)
    save_image(m, "matches")

Which gives you:

![matches](./figs/matches.jpg)


## 3.7 Fitting our projection to the data ##

Now that we have some matches we need to predict the projection between these two sets of points! However, this can be hard because we have a lot of noisy matches. Many of them are correct but we also have some outliers hiding in the data.

### 3.7.1 Projecting points with a homography ###

Implement `point project_point(matrix H, point p)` to project a point using matrix `H`. You can do this with the provided matrix library by calling `matrix_mult_matrix` (see `src/matrix.c`). Or you could pull out elements of the matrix and do the math yourself. Whatever you want! Just remember to do the proper normalization for converting from homogeneous coordinates back to image coordinates. (We talked about this in class, something with that w squiggly thing).

### 3.7.2  Calculate distances between points ###

`float point_distance(point p, point q)`. L2 distance. You know the formula.

### 3.7.3 Calculate model inliers ###

Figure out how many matches are inliers to a model. Fill in `int model_inliers(matrix H, match *m, int n, float thresh)` to loop over the points, project using the homography, and check if the projected point is within some `thresh`old distance of the actual matching point in the other image.

Also, we want to bring inliers to the front of the list. This helps us later on with the fitting functions. Again, this should all be doable in one pass through the data.

### 3.7.4 Fitting the homography ###

We will solve for the homography using the matrix operations discussed in class to solve equations like `M a = b`. Most of this is already implemented, you just have to fill in the matrices `M` and `b` with our match information in the first TODO in `matrix compute_homography(match *matches, int n)`.

You also have to read out the final results and populate our homography matrix. Consult the slides for details about what should go where, or derive it for yourself using the projection equations!

### 3.8 Randomize the matches ###

One of the steps in RANSAC is drawing random matches to estimate a new model. One easy way to do this is randomly shuffle the array of matches and then take the first `n` elements to fit a model.

Implement the [Fisher-Yates shuffle](https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle#The_modern_algorithm) in `void randomize_matches(match *m, int n)`.


## 3.9 Implement RANSAC ##

Implement the RANSAC algorithm discussed in class in `matrix RANSAC(match *m, int n, float thresh, int k, int cutoff)`. Pseudocode is provided.

## 3.10 Combine the images with a homography ##

Now we have to stitch the images together with our homography! Given two images and a homography, stitch them together with `image combine_images(image a, image b, matrix H)`.

Some of this is already filled in. The first step is to figure out the bounds to the image. To do this we'll project the corners of `b` back onto the coordinate system for `a` using `Hinv`. Then we can figure out our "new" coordinates and where to put `image a` on the canvas. Paste `a` in the appropriate spot in the canvas.

Next we need to loop over pixels that might map to `image b`, perform the mapping to see if they do, and if so fill in the pixels with the appropriate color from `b`. Our mapping will likely land between pixel values in `b` so use bilinear interpolation to compute the pixel value (good thing you already implemented this!).

With all this working you should be able to create some basic panoramas:

    im1 = load_image("data/Rainier1.png")
    im2 = load_image("data/Rainier2.png")
    pan = panorama_image(im1, im2, thresh=50)
    save_image(pan, "easy_panorama")

![panorama](./figs/easy_panorama.jpg)

Try out some of the other panorama creation in `tryhw3.py`. If you stitch together multiple images you should turn off the `if` statement in `panorama_image` that marks corners and draws inliers.

## Extra Credit ##

Mapping all the images back to the same coordinates is bad for large field-of-view panoramas, as discussed in class. Implement `image cylindrical_project(image im, float f)` to project an image to cylindrical coordinates and then unroll it. Then stitch together some very big panoramas. Send in your favorite. Use your own images if you want!

## More Extra Credit ##

Map images to spherical coordinates and stitch together a really big panorama! Will parts of it be upside-down??


## Turn it in ##

Turn in your `harris_image.c`, `panorama_image.c`, and some good panoramas you generated on canvas under Assignment 3.


# Homework 4 #

Welcome friends,

It's time for optical flow!

## 4.1 Integral images ##

Optical flow has to run on video so it needs to be fast! We'll be calculating structure matrices again so we need to do aggregated sums over regions of the image. However, instead of smoothing with a Gaussian filter, we'll use [integral images](https://en.wikipedia.org/wiki/Summed-area_table) to simulate smoothing with a box filter.

Fill in `make_integral_image` as described in the wikipedia article. You may not want to use the `get_pixel` methods unless you do bounds checking so you don't run into trouble.

## 4.2 Smoothing using integral images ##

We can use our integral image to quickly calculate the sum of regions of an image. Fill in `box_filter_image` so that every pixel in the output is the average of pixels in a given window size.

## 4.3 Lucas-Kanade optical flow ##

We'll be implementing [Lucas-Kanade](https://en.wikipedia.org/wiki/Lucas%E2%80%93Kanade_method) optical flow. We'll use a structure matrix but this time with temporal information as well. The equation we'll use is:

![](./figs/flow-eq.png)

### 4.3.1 Time-structure matrix ###

We'll need spatial and temporal gradient information for the flow equations. Calculate a time-structure matrix. Spatial gradients can be calculated as normal. The time gradient can be calculated as the difference between the previous image and the next image in a sequence. Fill in `time_structure_matrix`.

### 4.3.2 Calculating velocity from the structure matrix ###

Fill in `velocity_image` to use the equation to calculate the velocity of each pixel in the x and y direction. For each pixel, fill in the `matrix M`, invert it, and use it to calculate the velocity.

Try calculating the optical flow between two dog images:

    a = load_image("data/dog_a.jpg")
    b = load_image("data/dog_b.jpg")
    flow = optical_flow_images(b, a, 15, 8)
    draw_flow(a, flow, 8)
    save_image(a, "lines")

It may look something like:

![](./figs/lines.png)

## 4.4 Optical flow demo using OpenCV ## 

Using OpenCV we can get images from the webcam and display the results in real-time. Try installing OpenCV and enabling OpenCV compilation in the Makefile (set `OPENCV=1`). Then run:

    optical_flow_webcam(15,4,8)


## Turn it in ##

Turn in your `flow_image.c` on canvas under Assignment 4.

# Homework 5 #

Welcome friends,

It's time for neural networks!

## 5.1 Implementing neural networks ##

### 5.1.1 Activation functions ###

An important part of machine learning, be it linear classifiers or neural networks, is the activation function you use. Fill in `void activate_matrix(matrix m, ACTIVATION a)` to modify `m` to be `f(m)` applied elementwise where the function `f` is given by what the activation `a` is.

Remember that for our softmax activation we will take e^x for every element x, but then we have to normalize each element by the sum as well. Each row in the matrix is a separate data point so we want to normalize over each data point separately.

### 5.1.2 Taking gradients ###

As we are calculating the partial derivatives for the weights of our model we have to remember to factor in the gradient from our activation function at every layer. We will have the derivative of the loss with respect to the output of a layer (after activation) and we need the derivative of the loss with respect to the input (before activation). To do that we take each element in our delta (the partial derivative) and multiply it by the gradient of our activation function.

Normally, to calculate `f'(x)` we would need to remember what the input to the function, `x`, was. However, all of our activation functions have the nice property that we can calculate the gradient `f'(x)` only with the output `f(x)`. This means we have to remember less stuff when running our model.

The gradient of a linear activation is just 1 everywhere. The gradient of our softmax will also be 1 everywhere because we will only use the softmax as our output with a cross-entropy loss function, for an explaination of why see [here](https://math.stackexchange.com/questions/945871/derivative-of-softmax-loss-function?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa) or [here](https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/#softmax-and-cross-entropy-loss).

The gradient of our logistic function is discussed [here](https://en.wikipedia.org/wiki/Logistic_function#Derivative):

    f'(x) = f(x) * (1 - f(x))

I'll let you figure out on your own what `f'(x)` given `f(x)` is for RELU and Leaky RELU.

As discussed in the backpropagation slides, we just multiply our partial derivative by `f'(x)` to backpropagate through an activation function. Fill in `void gradient_matrix(matrix m, ACTIVATION a, matrix d)` to multiply the elements of `d` by the correct gradient information where `m` is the output of a layer that uses `a` as it's activation function.

### 5.1.3 Forward propagation ###

Now we can fill in how to forward-propagate information through a layer. First check out our layer struct:

    typedef struct {
        matrix in;              // Saved input to a layer
        matrix w;               // Current weights for a layer
        matrix dw;              // Current weight updates
        matrix v;               // Past weight updates (for use with momentum)
        matrix out;             // Saved output from the layer
        ACTIVATION activation;  // Activation the layer uses
    } layer;

During forward propagation we will do a few things. We'll multiply the input matrix by our weight matrix `w`. We'll apply the activation function `activation`. We'll also save the input and output matrices in the layer so that we can use them during backpropagation. But this is already done for you.

Fill in the TODO in `matrix forward_layer(layer *l, matrix in)`. You will need the matrix multiplication function `matrix matrix_mult_matrix(matrix a, matrix b)` provided by our matrix library.

Using matrix operations we can batch-process data. Each row in the input `in` is a separate data point and each row in the returned matrix is the result of passing that data point through the layer.

### 5.1.4 Backward propagation ###

We have to backpropagate error through our layer. We will be given `dL/dy`, the derivative of the loss with respect to the output of the layer, `y`. The output `y` is given by `y = f(xw)` where `x` is the input, `w` is our weights for that layer, and `f` is the activation function. What we want to calculate is `dL/dw` so we can update our weights and `dL/dx` which is the backpropagated loss to the previous layer. You'll be filling in `matrix backward_layer(layer *l, matrix delta)`

#### 5.1.4.1 Gradient of activation function ####

First we need to calculate `dL/d(xw)` using the gradient of our activation function. Recall:

    dL/d(xw) = dL/dy * dy/d(xw)
             = dL/dy * df(xw)/d(xw)
             = dL/dy * f'(xw)

Use your `void gradient_matrix(matrix m, ACTIVATION a, matrix d)` function to change delta from `dL/dy` to `dL/d(xw)`

#### 5.1.4.2 Derivative of loss w.r.t. weights ####

Next we want to calculate the derivative with respect to our weights, `dL/dw`. Recall:

    dL/dw = dL/d(xw) * d(xw)/dw
          = dL/d(xw) * x

but remember from the slides, to make the matrix dimensions work out right we acutally do the matrix operiation of `xt * dL/d(xw)` where `xt` is the transpose of the input matrix `x`.

In our layer we saved the input as `l->in`. Calculate `xt` using that and the matrix transpose function in our library, `matrix transpose_matrix(matrix m)`. Then calculate `dL/dw` and save it into `l->dw` (free the old one first to not leak memory!). We'll use this later when updating our weights.

#### 5.1.4.3 Derivative of loss w.r.t. input ####

Next we want to calculate the derivative with respect to the input as well, `dL/dx`. Recall:

    dL/dx = dL/d(xw) * d(xw)/dx
          = dL/d(xw) * w

again, we have to make the matrix dimensions line up so it actually ends up being `dL/d(xw) * wt` where `wt` is the transpose of our weights, `w`. Calculate `wt` and then calculate dL/dx. This is the matrix we will return.

### 5.1.5 Weight updates ###

After we've performed a round of forward and backward propagation we want to update our weights. We'll fill in `void update_layer(layer *l, double rate, double momentum, double decay)` to do just that.

Remember from class with momentum and decay our weight update rule is:

    Δw_t = dL/dw_t - λw_t + mΔw_{t-1}
    w_{t+1} = w_t + ηΔw_t

We'll be doing gradient ascent (the partial derivative component is positive) instead of descent because we'll flip the sign of our partial derivative when we calculate it at the output. We saved `dL/dw_t` as `l->dw` and by convention we'll use `l->v` to store the previous weight change `Δw_{t-1}`.

Calculate the current weight change as a weighted sum of `l->dw`, `l->w`, and `l->v`. The function `matrix axpy_matrix(double a, matrix x, matrix y)` will be useful here, it calculates the result of the operation `ax+y` where `a` is a scalar and `x` and `y` are matrices. Save the current weight change in `l->v` for next round (remember to free the current `l->v` first).

Finally, apply the weight change to your weights by adding a scaled amount of the change based on the learning rate.

## 5.1.6 Read through the rest of the code ##

Check out the remainder of the functions which string layers together to make a model, run the model forward and backward, update it, train it, etc. Notable highlights:

#### `layer make_layer(int input, int output, ACTIVATION activation)` ####

Makes a new layer that takes `input` number of inputs and produces `output` number of outputs. Note that we randomly initialize the weight vector, in this case with a uniform random distribution between `[-sqrt(2/input), sqrt(2/input)]`. The weights could be initialize with other random distributions but this one works pretty well. This is one of those secret tricks you just have to know from experience!

#### `double accuracy_model(model m, data d)` ####

Will be useful to test out our model once we are done training it.


#### `double cross_entropy_loss(matrix y, matrix p)` ####

Calculates the cross-entropy loss for the ground-truth labels `y` and our predictions `p`. Cross-entropy loss is just negative log-likelihood (we discussed log-likelihood in class for logistic regression) for multi-class classification. Since we added the negative sign it becomes a loss function that we want to minimize. Cross-entropy loss is given by:

    L = Σ(-y_i log(p_i))

for a single data point, summing across the different classes. Cross-entropy loss is nice to use with our softmax activation because the partial derivatives are nice. If the output of our final layer uses a softmax: `y = σ(wx)` then:

    dL/d(wx) = (y - truth)

During training, we'll just calculate `dL/dy` as `(y - truth)` and set the gradient of the softmax to be 1 everywhere to have the same effect, as discussed earlier. This avoids numerical instability with calculating the "true" `dL/dy` and `dσ(x)/dx` for cross-entropy loss and the softmax function independently and multiplying them together.

#### `void train_model(...)` ####

Our training code to implement SGD. First we get a random subset of the data using `data random_batch(data d, int n)`. Then we run our model forward and calculate the error we make `dL/dy`. Finally we backpropagate the error through the model and update the weights at every layer.

## 5.2 Experiments with MNIST ##

We'll be testing out your implementation on the [MNIST dataset](http://yann.lecun.com/exdb/mnist/)!

When there are questions during this section please answer them in the `tryhw5.py` file, you will submit this file in Canvas as well as `classifier.c`. Answers can be short, 1-2 sentences.

### 5.2.1 Getting the data ###

To run your model you'll need the dataset. The training images can be found [here](https://pjreddie.com/media/files/mnist_train.tar.gz) and the test images are [here](https://pjreddie.com/media/files/mnist_test.tar.gz), I've preprocessed them for you into PNG format. To get the data you can run:

    wget https://pjreddie.com/media/files/mnist_train.tar.gz
    wget https://pjreddie.com/media/files/mnist_test.tar.gz
    tar xzf mnist_train.tar.gz
    tar xzf mnist_test.tar.gz

We'll also need a list of the images in our training and test set. To do this you can run:

    find train -name \*.png > mnist.train
    find test -name \*.png > mnist.test

Or something similar.

### 5.2.2 Train a linear softmax model ###

Check out `tryhw5.py` to see how we're actually going to run the machine learning code you wrote. There are a couple example models in here for softmax regression and neural networks. Run the file with: `python tryhw5.py` to train a softmax regression model on MNIST. You will see a bunch of numbers scrolling by, this is the loss calculated by the model for the current batch. Hopefully over time this loss goes down as the model improves.

After training our python script will calculate the accuracy the model gets on both the training dataset and a separate testing dataset.

#### 5.2.2.1 Question ####

Why might we be interested in both training accuracy and testing accuracy? What do these two numbers tell us about our current model?

#### 5.2.2.2 Question ####

Try varying the model parameter for learning rate to different powers of 10 (i.e. 10^1, 10^0, 10^-1, 10^-2, 10^-3) and training the model. What patterns do you see and how does the choice of learning rate affect both the loss during training and the final model accuracy?

#### 5.2.2.3 Question ####

Try varying the parameter for weight decay to different powers of 10: (10^0, 10^-1, 10^-2, 10^-3, 10^-4, 10^-5). How does weight decay affect the final model training and test accuracy?

### 5.2.3 Train a neural network ###

Now change the training code to use the neural network model instead of the softmax model.

#### 5.2.3.1 Question ####

Currently the model uses a logistic activation for the first layer. Try using a the different activation functions we programmed. How well do they perform? What's best?

#### 5.2.3.2 Question ####

Using the same activation, find the best (power of 10) learning rate for your model. What is the training accuracy and testing accuracy?

#### 5.2.3.3 Question ####

Right now the regularization parameter `decay` is set to 0. Try adding some decay to your model. What happens, does it help? Why or why not may this be?


#### 5.2.3.4 Question ####

Modify your model so it has 3 layers instead of two. The layers should be `inputs -> 64`, `64 -> 32`, and `32 -> outputs`. Also modify your model to train for 3000 iterations instead of 1000. Look at the training and testing accuracy for different values of decay (powers of 10, 10^-4 -> 10^0). Which is best? Why?

## 5.3 Training on CIFAR ##

The [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset is meant to be similar to MNIST but much more challenging. Using the same model we just created, let's try training on CIFAR.

### 5.3.1 Get the data ###

We have to do a similar process as last time, getting the data and creating files to hold the paths. Run:

    wget http://pjreddie.com/media/files/cifar.tgz
    tar xzf cifar.tgz
    find cifar/train -name \*.png > cifar.train
    find cifar/test -name \*.png > cifar.test

Notice that the file of possible labels can be found in `cifar/labels.txt`

### 5.3.2 Train on CIFAR ###

Modify `tryhw5.py` to use CIFAR. This should mostly involve swapping out references to `mnist` with `cifar` during dataset loading. Then try training the network. You may have to fiddle with the learning rate to get it to train well.

#### 5.3.2.1 Question ####

How well does your network perform on the CIFAR dataset?

## Turn it in ##

Turn in your `classifier.c` on canvas under Assignment 5.
