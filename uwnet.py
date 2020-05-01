import sys, os
from ctypes import *
import math
import random

lib = CDLL(os.path.join(os.path.dirname(__file__), "libuwimg.so"), RTLD_GLOBAL)

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]
    def __add__(self, other):
        return add_image(self, other,1)
    def __sub__(self, other):
        return add_image(self, other,0)

class POINT(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float)]

class DESCRIPTOR(Structure):
    _fields_ = [("p", POINT),
                ("n", c_int),
                ("data", POINTER(c_float))]

class MATRIX(Structure):
    _fields_ = [("rows", c_int),
                ("cols", c_int),
                ("data", POINTER(c_float)),
                ("shallow", c_int)]

class DATA(Structure):
    _fields_ = [("X", MATRIX),
                ("y", MATRIX)]

class LAYER(Structure):
    pass

LAYER._fields_ = [("in",  POINTER(MATRIX)),
                ("out",   POINTER(MATRIX)),
                ("delta", POINTER(MATRIX)),
                ("w", MATRIX),
                ("dw", MATRIX),
                ("b", MATRIX),
                ("db", MATRIX),
                ("width", c_int),
                ("height", c_int),
                ("channels", c_int),
                ("size", c_int),
                ("stride", c_int),
                ("filters", c_int),
                ("activation", c_int),
                ("batchnorm", c_int),
                ("x_norm", MATRIX),
                ("rolling_mean", MATRIX),
                ("rolling_variance", MATRIX),
                ("x", POINTER(MATRIX)),
                ("forward", CFUNCTYPE(MATRIX, POINTER(LAYER), MATRIX)),
                ("backward", CFUNCTYPE(None, POINTER(LAYER), MATRIX)),
                ("update", CFUNCTYPE(None, POINTER(LAYER), c_float, c_float, c_float))]

class NET(Structure):
    _fields_ = [("layers", POINTER(LAYER)),
                ("n", c_int)]


(LINEAR, LOGISTIC, RELU, LRELU, SOFTMAX) = range(5)


add_image = lib.add_image
add_image.argtypes = [IMAGE,IMAGE]
add_image.restype = IMAGE

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

free_image = lib.free_image
free_image.argtypes = [POINTER(IMAGE)]

get_pixel = lib.get_pixel
get_pixel.argtypes = [IMAGE, c_int, c_int, c_int]
get_pixel.restype = c_float

set_pixel = lib.set_pixel
set_pixel.argtypes = [IMAGE, c_int, c_int, c_int, c_float]
set_pixel.restype = None

rgb_to_grayscale = lib.rgb_to_grayscale
rgb_to_grayscale.argtypes = [IMAGE]
rgb_to_grayscale.restype = IMAGE

copy_image = lib.copy_image
copy_image.argtypes = [IMAGE]
copy_image.restype = IMAGE

rgb_to_hsv = lib.rgb_to_hsv
rgb_to_hsv.argtypes = [IMAGE]
rgb_to_hsv.restype = None

feature_normalize = lib.feature_normalize
feature_normalize.argtypes = [IMAGE]
feature_normalize.restype = None

clamp_image = lib.clamp_image
clamp_image.argtypes = [IMAGE]
clamp_image.restype = None

hsv_to_rgb = lib.hsv_to_rgb
hsv_to_rgb.argtypes = [IMAGE]
hsv_to_rgb.restype = None

shift_image = lib.shift_image
shift_image.argtypes = [IMAGE, c_int, c_float]
shift_image.restype = None

load_image_lib = lib.load_image
load_image_lib.argtypes = [c_char_p]
load_image_lib.restype = IMAGE

def load_image(f):
    return load_image_lib(f.encode('utf-8'))

# Filetypes
(PNG, BMP, TGA, JPG) = range(4)
(BOX,HPASS,SHARPEN,EMBOSS,GAUSS,GX,GY)=range(7)

save_image_options_lib = lib.save_image_options
save_image_options_lib.argtypes = [IMAGE, c_char_p, c_int, c_int]
save_image_options_lib.restype = None

def save_image(im, f):
    return save_image_options_lib(im, f.encode('utf-8'), JPG, 80)

def save_png(im, f):
    return save_image_options_lib(im, f.encode('utf-8'), PNG, 0)

same_image = lib.same_image
same_image.argtypes = [IMAGE,IMAGE]
same_image.restype = c_int

resize = lib.resize
resize.argtypes = [IMAGE, c_int, c_int,c_int]
resize.restype = IMAGE


make_filter = lib.make_filter
make_filter.argtypes = [c_int,c_float]
make_filter.restype = IMAGE

sobel_image = lib.sobel_image
sobel_image.argtypes = [IMAGE]
sobel_image.restype = POINTER(IMAGE)

colorize_sobel = lib.colorize_sobel
colorize_sobel.argtypes = [IMAGE]
colorize_sobel.restype = IMAGE

convolve_image = lib.convolve_image
convolve_image.argtypes = [IMAGE,IMAGE, c_int]
convolve_image.restype = IMAGE

harris_corner_detector = lib.harris_corner_detector
harris_corner_detector.argtypes = [IMAGE, c_float, c_float, c_int, POINTER(c_int)]
harris_corner_detector.restype = POINTER(DESCRIPTOR)

mark_corners = lib.mark_corners
mark_corners.argtypes = [IMAGE, IMAGE, c_int]
mark_corners.restype = None

detect_and_draw_corners = lib.detect_and_draw_corners
detect_and_draw_corners.argtypes = [IMAGE, c_float, c_float, c_int]
detect_and_draw_corners.restype = None

cylindrical_project = lib.cylindrical_project
cylindrical_project.argtypes = [IMAGE, c_float]
cylindrical_project.restype = IMAGE

structure_matrix = lib.structure_matrix
structure_matrix.argtypes = [IMAGE, c_float]
structure_matrix.restype = IMAGE

find_and_draw_matches = lib.find_and_draw_matches
find_and_draw_matches.argtypes = [IMAGE, IMAGE, c_float, c_float, c_int]
find_and_draw_matches.restype = IMAGE

panorama_image_lib = lib.panorama_image
panorama_image_lib.argtypes = [IMAGE,IMAGE, c_float, c_float, c_int, c_float, c_int, c_int]
panorama_image_lib.restype = IMAGE

draw_flow = lib.draw_flow
draw_flow.argtypes = [IMAGE,IMAGE, c_float]
draw_flow.restype = None

box_filter_image = lib.box_filter_image
box_filter_image.argtypes = [IMAGE, c_int]
box_filter_image.restype = IMAGE

optical_flow_images = lib.optical_flow_images
optical_flow_images.argtypes = [IMAGE,IMAGE, c_int, c_int]
optical_flow_images.restype = IMAGE

optical_flow_webcam = lib.optical_flow_webcam
optical_flow_webcam.argtypes = [c_int, c_int, c_int,c_char_p]
optical_flow_webcam.restype = None

def panorama_image(a, b, sigma=2, thresh=5, nms=3, inlier_thresh=2, iters=10000, cutoff=30):
    return panorama_image_lib(a, b, sigma, thresh, nms, inlier_thresh, iters, cutoff)


train_image_classifier = lib.train_image_classifier
train_image_classifier.argtypes = [NET, DATA, c_int, c_int, c_float, c_float, c_float]
train_image_classifier.restype = None

accuracy_net = lib.accuracy_net
accuracy_net.argtypes = [NET, DATA]
accuracy_net.restype = c_float

forward_net = lib.forward_net
forward_net.argtypes = [NET, MATRIX]
forward_net.restype = MATRIX

load_image_classification_data_lib = lib.load_image_classification_data
load_image_classification_data_lib.argtypes = [c_char_p, c_char_p]
load_image_classification_data_lib.restype = DATA

def load_image_classification_data(images, labels):
    return load_image_classification_data_lib(images.encode('utf-8'), labels.encode('utf-8'))

make_connected_layer_lib = lib.make_connected_layer
make_connected_layer_lib.argtypes = [c_int, c_int, c_int]
make_connected_layer_lib.restype = LAYER

def make_connected_layer(inputs, outputs, activation, batchnorm = 0):
    l = make_connected_layer_lib(inputs, outputs, activation)
    l.batchnorm = batchnorm
    return l
    
make_convolutional_layer_lib = lib.make_convolutional_layer
make_convolutional_layer_lib.argtypes = [c_int, c_int, c_int, c_int, c_int, c_int, c_int]
make_convolutional_layer_lib.restype = LAYER

def make_convolutional_layer(w, h, c, filters, size, stride, activation, batchnorm = 0):
    l = make_convolutional_layer_lib(w, h, c, filters, size, stride, activation)
    l.batchnorm = batchnorm
    return l

make_maxpool_layer = lib.make_maxpool_layer
make_maxpool_layer.argtypes = [c_int, c_int, c_int, c_int, c_int]
make_maxpool_layer.restype = LAYER

clear_net = lib.clear_net
clear_net.argtypes = [NET]
clear_net.restype = None

save_weights_lib = lib.save_weights
save_weights_lib.argtypes = [NET, c_char_p]
save_weights_lib.restype = None

load_weights_lib = lib.load_weights
load_weights_lib.argtypes = [NET, c_char_p]
load_weights_lib.restype = None

def save_weights(net, f):
    save_weights_lib(net, f.encode('utf-8'))

def load_weights(net, f):
    load_weights_lib(net, f.encode('utf-8'))

print_matrix = lib.print_matrix
print_matrix.argtypes = [MATRIX]
print_matrix.restype = None

def run_net_image(net, im):
    m = MATRIX()
    m.rows = 1
    m.cols = im.h*im.w*im.c
    m.data = im.data
    m.shallow = 1
    return forward_net(net, m)

def make_net(layers):
    m = NET()
    m.n = len(layers)
    m.layers = (LAYER*m.n) (*layers)
    return m

if __name__ == "__main__":
    im = load_image("data/dog.jpg")
    save_image(im, "hey")

