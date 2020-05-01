#include <math.h>
#include <string.h>
#include <stdlib.h>
#include "uwnet.h"
#include "image.h"
#include "test.h"
#include "args.h"
void try_mnist()
{
    data train = load_image_classification_data("mnist/mnist.train", "mnist/mnist.labels");
    data test  = load_image_classification_data("mnist/mnist.test", "mnist/mnist.labels");

    net n = {0};
    n.layers = calloc(3, sizeof(layer));
    n.n = 3;
    n.layers[0] = make_convolutional_layer(28, 28, 1, 1, 5, 2, LRELU);
    n.layers[1] = make_convolutional_layer(14, 14, 1, 8, 5, 2, LRELU);
    n.layers[2] = make_connected_layer(392, 10, SOFTMAX);
	n.layers[0].batchnorm=1;
	n.layers[1].batchnorm=1;
	n.layers[2].batchnorm=1;
    int batch = 128;
    int iters = 5000;
    float rate = .01;
    float momentum = .9;
    float decay = .0005;

    train_image_classifier(n, train, batch, iters, rate, momentum, decay);
    printf("Training accuracy: %f\n", accuracy_net(n, train));
    printf("Testing  accuracy: %f\n", accuracy_net(n, test));
}

int main(int argc, char **argv)
{
  	if (0 == strcmp(argv[1], "test"))run_tests();
    else if (0 == strcmp(argv[1], "hw0")) test_hw0();
    else if (0 == strcmp(argv[1], "hw1")) test_hw1();
    else if (0 == strcmp(argv[1], "hw2")) test_hw2();
    else if (0 == strcmp(argv[1], "hw3")) test_hw3();
    else if (0 == strcmp(argv[1], "hw4")) test_hw4();
	else if (0 == strcmp(argv[1], "mnist")) try_mnist();
	else {
		printf("usage: %s <test | hw0 | hw1...>\n", argv[0]);
#ifdef OPENCV
		printf("optical_flow by %s", argv[1] ? argv[1] : "usbcam");
		optical_flow_webcam(15, 2, 4, argv[1]);
#endif
	}
    return 0;
}
