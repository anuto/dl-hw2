from uwnet import *

def neural_net():
    l =[ 
            make_connected_layer(3072, 1500, LRELU),
            make_connected_layer(1500, 500, LRELU),
            make_connected_layer(500, 256, LRELU),
            make_connected_layer(256, 10, SOFTMAX)]
    return make_net(l)

def conv_net():
    l = [   make_convolutional_layer(32, 32, 3, 8, 3, 1, LRELU, 1),
            make_maxpool_layer(32, 32, 8, 3, 2),
            make_convolutional_layer(16, 16, 8, 16, 3, 1, LRELU, 1),
            make_maxpool_layer(16, 16, 16, 3, 2),
            make_convolutional_layer(8, 8, 16, 32, 3, 1, LRELU, 1),
            make_maxpool_layer(8, 8, 32, 3, 2),
            make_convolutional_layer(4, 4, 32, 64, 3, 1, LRELU, 1),
            make_maxpool_layer(4, 4, 64, 3, 2),
            make_connected_layer(256, 10, SOFTMAX)]
    return make_net(l)

print("loading data...")
train = load_image_classification_data("cifar/cifar.train", "cifar/cifar.labels")
test  = load_image_classification_data("cifar/cifar.test",  "cifar/cifar.labels")
print("done")
print

print("making model...")
batch = 128
iters = 3000
rate = .01
momentum = .9
decay = .005

# Our best result: 
# .1 learning rate
# 5000 epochs
# evaluating model...
# ('training accuracy: %f', 0.741919994354248)
# ('test accuracy:     %f', 0.6972000002861023)
# Surprisingly, there was no obvious trend when changing the learning rates. 
# All variations on the learning rate parameter capped out at about 60-70% accuracy
# and about the same loss of 1.0

m = conv_net()
print("training...")
train_image_classifier(m, train, batch, iters, rate, momentum, decay)
print("done")
print

print("evaluating model...")
print("training accuracy: %f", accuracy_net(m, train))
print("test accuracy:     %f", accuracy_net(m, test))

