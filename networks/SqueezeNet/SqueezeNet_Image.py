import tensorflow as tf
import numpy as np
import sys
import skimage
import skimage.io
import skimage.transform

# Parameters
learning_rate = 0.01
training_epochs = 25
batch_size = 100
display_step = 1


class SqueezeNet(object):
    def __init__(self, nb_classes=2, is_training=True):
        # conv1
        self.X_batch = tf.placeholder(tf.float32,shape=[None,224,224,3])
        self.Y_batch = tf.placeholder(tf.float32,shape=[None, 2])

        net = tf.layers.conv2d(self.X_batch, 96, [7, 7], strides=[2, 2],
                                 padding="SAME", activation=tf.nn.relu,
                                 name="conv1")
        # maxpool1
        net = tf.layers.max_pooling2d(net, [3, 3], strides=[2, 2],
                                      name="maxpool1")
        # fire2
        net = self._fire(net, 16, 64, "fire2")
        # fire3
        net = self._fire(net, 16, 64, "fire3")
        # fire4
        net = self._fire(net, 32, 128, "fire4")
        # maxpool4
        net = tf.layers.max_pooling2d(net, [3, 3], strides=[2, 2],
                                      name="maxpool4")
        # fire5
        net = self._fire(net, 32, 128, "fire5")
        # fire6
        net = self._fire(net, 48, 192, "fire6")
        # fire7
        net = self._fire(net, 48, 192, "fire7")
        # fire8
        net = self._fire(net, 64, 256, "fire8")
        # maxpool8
        net = tf.layers.max_pooling2d(net, [3, 3], strides=[2, 2],
                                      name="maxpool8")
        # fire9
        net = self._fire(net, 64, 256, "fire9")
        # dropout
        net = tf.layers.dropout(net, 0.5, training=is_training)
        # conv10
        # CURRENTLY 2 CLASS
        net = tf.layers.conv2d(net, 2, [1, 1], strides=[1, 1],
                               padding="SAME", activation=tf.nn.relu,
                               name="conv10")
        # avgpool10
        net = tf.layers.average_pooling2d(net, [13, 13], strides=[1, 1],
                                          name="avgpool10")
        # squeeze the axis
        net = tf.squeeze(net, axis=[1, 2])

        self.logits = net
        self.prediction = tf.nn.softmax(net)

        self.losses = tf.losses.softmax_cross_entropy(onehot_labels=self.Y_batch,logits=self.logits)
        self.train = tf.train.GradientDescentOptimizer(0.1).minimize(self.losses)


    def _fire(self, inputs, squeeze_depth, expand_depth, scope):
        with tf.variable_scope(scope):
            squeeze = tf.layers.conv2d(inputs, squeeze_depth, [1, 1],
                                       strides=[1, 1], padding="SAME",
                                       activation=tf.nn.relu, name="squeeze")
            # squeeze
            expand_1x1 = tf.layers.conv2d(squeeze, expand_depth, [1, 1],
                                          strides=[1, 1], padding="SAME",
                                          activation=tf.nn.relu, name="expand_1x1")
            expand_3x3 = tf.layers.conv2d(squeeze, expand_depth, [3, 3],
                                          strides=[1, 1], padding="SAME",
                                          activation=tf.nn.relu, name="expand_3x3")
            return tf.concat([expand_1x1, expand_3x3], axis=3)



def load_image(path):
    # load image
    img = skimage.io.imread(path)
    img = img / 255.0
    # assert (0 <= img).all() and (img <= 1.0).all()
    # print "Original Image Shape: ", img.shape
    # we crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    # resize to 224, 224
    resized_img = skimage.transform.resize(crop_img, (224, 224))
    return resized_img


if __name__ == "__main__":
    path = sys.path[0]
    img1 = load_image(path+"/dog.jpg")*255.0
    img2 = load_image(path+"/cat.jpg")*255.0
    batch1 = img1.reshape((1, 224, 224, 3))
    batch2 = img2.reshape((1, 224, 224, 3))
    x = np.concatenate((batch1, batch2), 0)
    y = np.array([[1, 0],[0, 1]], dtype=np.int64)
    with tf.Graph().as_default():
        with tf.Session() as sess:
            SqueezeNet = SqueezeNet(x)
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())
            _, loss = sess.run([SqueezeNet.train, SqueezeNet.losses], feed_dict={SqueezeNet.X_batch: x, SqueezeNet.Y_batch: y})

            saver.save(sess, "saved_model/model-32")
            tf.summary.FileWriter("saved_model", sess.graph)
            print(loss)
