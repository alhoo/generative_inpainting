#!/usr/bin/env python


import math
from gimpfu import *

import cv2
import numpy as np
import tensorflow as tf
import neuralgym as ng

#from inpaint_model import InpaintCAModel

""" common model for DCGAN """
import logging

import cv2
import neuralgym as ng
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import arg_scope

from neuralgym.models import Model
from neuralgym.ops.summary_ops import scalar_summary, images_summary
from neuralgym.ops.summary_ops import gradients_summary
from neuralgym.ops.layers import flatten, resize
from neuralgym.ops.gan_ops import gan_wgan_loss, gradients_penalty
from neuralgym.ops.gan_ops import random_interpolates

#from inpaint_ops import gen_conv, gen_deconv, dis_conv
#from inpaint_ops import random_bbox, bbox2mask, local_patch
#from inpaint_ops import spatial_discounting_mask
#from inpaint_ops import resize_mask_like, contextual_attention

import logging

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import add_arg_scope

from neuralgym.ops.layers import resize
from neuralgym.ops.layers import *
from neuralgym.ops.loss_ops import *
from neuralgym.ops.summary_ops import *


logger = logging.getLogger()
np.random.seed(2018)


@add_arg_scope
def gen_conv(x, cnum, ksize, stride=1, rate=1, name='conv',
             padding='SAME', activation=tf.nn.elu, training=True):
    """Define conv for generator.

    Args:
        x: Input.
        cnum: Channel number.
        ksize: Kernel size.
        Stride: Convolution stride.
        Rate: Rate for or dilated conv.
        name: Name of layers.
        padding: Default to SYMMETRIC.
        activation: Activation function after convolution.
        training: If current graph is for training or inference, used for bn.

    Returns:
        tf.Tensor: output

    """
    assert padding in ['SYMMETRIC', 'SAME', 'REFELECT']
    if padding == 'SYMMETRIC' or padding == 'REFELECT':
        p = int(rate*(ksize-1)/2)
        x = tf.pad(x, [[0,0], [p, p], [p, p], [0,0]], mode=padding)
        padding = 'VALID'
    x = tf.layers.conv2d(
        x, cnum, ksize, stride, dilation_rate=rate,
        activation=activation, padding=padding, name=name)
    return x


@add_arg_scope
def gen_deconv(x, cnum, name='upsample', padding='SAME', training=True):
    """Define deconv for generator.
    The deconv is defined to be a x2 resize_nearest_neighbor operation with
    additional gen_conv operation.

    Args:
        x: Input.
        cnum: Channel number.
        name: Name of layers.
        training: If current graph is for training or inference, used for bn.

    Returns:
        tf.Tensor: output

    """
    with tf.variable_scope(name):
        x = resize(x, func=tf.image.resize_nearest_neighbor)
        x = gen_conv(
            x, cnum, 3, 1, name=name+'_conv', padding=padding,
            training=training)
    return x


@add_arg_scope
def dis_conv(x, cnum, ksize=5, stride=2, name='conv', training=True):
    """Define conv for discriminator.
    Activation is set to leaky_relu.

    Args:
        x: Input.
        cnum: Channel number.
        ksize: Kernel size.
        Stride: Convolution stride.
        name: Name of layers.
        training: If current graph is for training or inference, used for bn.

    Returns:
        tf.Tensor: output

    """
    x = tf.layers.conv2d(x, cnum, ksize, stride, 'SAME', name=name)
    x = tf.nn.leaky_relu(x)
    return x


def random_bbox(config):
    """Generate a random tlhw with configuration.

    Args:
        config: Config should have configuration including IMG_SHAPES,
            VERTICAL_MARGIN, HEIGHT, HORIZONTAL_MARGIN, WIDTH.

    Returns:
        tuple: (top, left, height, width)

    """
    img_shape = config.IMG_SHAPES
    img_height = img_shape[0]
    img_width = img_shape[1]
    maxt = img_height - config.VERTICAL_MARGIN - config.HEIGHT
    maxl = img_width - config.HORIZONTAL_MARGIN - config.WIDTH
    t = tf.random_uniform(
        [], minval=config.VERTICAL_MARGIN, maxval=maxt, dtype=tf.int32)
    l = tf.random_uniform(
        [], minval=config.HORIZONTAL_MARGIN, maxval=maxl, dtype=tf.int32)
    h = tf.constant(config.HEIGHT)
    w = tf.constant(config.WIDTH)
    return (t, l, h, w)


def bbox2mask(bbox, config, name='mask'):
    """Generate mask tensor from bbox.

    Args:
        bbox: configuration tuple, (top, left, height, width)
        config: Config should have configuration including IMG_SHAPES,
            MAX_DELTA_HEIGHT, MAX_DELTA_WIDTH.

    Returns:
        tf.Tensor: output with shape [1, H, W, 1]

    """
    def npmask(bbox, height, width, delta_h, delta_w):
        mask = np.zeros((1, height, width, 1), np.float32)
        h = np.random.randint(delta_h//2+1)
        w = np.random.randint(delta_w//2+1)
        mask[:, bbox[0]+h:bbox[0]+bbox[2]-h,
             bbox[1]+w:bbox[1]+bbox[3]-w, :] = 1.
        return mask
    with tf.variable_scope(name), tf.device('/cpu:0'):
        img_shape = config.IMG_SHAPES
        height = img_shape[0]
        width = img_shape[1]
        mask = tf.py_func(
            npmask,
            [bbox, height, width,
             config.MAX_DELTA_HEIGHT, config.MAX_DELTA_WIDTH],
            tf.float32, stateful=False)
        mask.set_shape([1] + [height, width] + [1])
    return mask


def local_patch(x, bbox):
    """Crop local patch according to bbox.

    Args:
        x: input
        bbox: (top, left, height, width)

    Returns:
        tf.Tensor: local patch

    """
    x = tf.image.crop_to_bounding_box(x, bbox[0], bbox[1], bbox[2], bbox[3])
    return x


def resize_mask_like(mask, x):
    """Resize mask like shape of x.

    Args:
        mask: Original mask.
        x: To shape of x.

    Returns:
        tf.Tensor: resized mask

    """
    mask_resize = resize(
        mask, to_shape=x.get_shape().as_list()[1:3],
        func=tf.image.resize_nearest_neighbor)
    return mask_resize


def spatial_discounting_mask(config):
    """Generate spatial discounting mask constant.

    Spatial discounting mask is first introduced in publication:
        Generative Image Inpainting with Contextual Attention, Yu et al.

    Args:
        config: Config should have configuration including HEIGHT, WIDTH,
            DISCOUNTED_MASK.

    Returns:
        tf.Tensor: spatial discounting mask

    """
    gamma = config.SPATIAL_DISCOUNTING_GAMMA
    shape = [1, config.HEIGHT, config.WIDTH, 1]
    if config.DISCOUNTED_MASK:
        logger.info('Use spatial discounting l1 loss.')
        mask_values = np.ones((config.HEIGHT, config.WIDTH))
        for i in range(config.HEIGHT):
            for j in range(config.WIDTH):
                mask_values[i, j] = max(
                    gamma**min(i, config.HEIGHT-i),
                    gamma**min(j, config.WIDTH-j))
        mask_values = np.expand_dims(mask_values, 0)
        mask_values = np.expand_dims(mask_values, 3)
        mask_values = mask_values
    else:
        mask_values = np.ones(shape)
    return tf.constant(mask_values, dtype=tf.float32, shape=shape)


def contextual_attention(f, b, mask=None, ksize=3, stride=1, rate=1,
                         fuse_k=3, softmax_scale=10., training=True, fuse=True):
    """ Contextual attention layer implementation.

    Contextual attention is first introduced in publication:
        Generative Image Inpainting with Contextual Attention, Yu et al.

    Args:
        x: Input feature to match (foreground).
        t: Input feature for match (background).
        mask: Input mask for t, indicating patches not available.
        ksize: Kernel size for contextual attention.
        stride: Stride for extracting patches from t.
        rate: Dilation for matching.
        softmax_scale: Scaled softmax for attention.
        training: Indicating if current graph is training or inference.

    Returns:
        tf.Tensor: output

    """
    # get shapes
    raw_fs = tf.shape(f)
    raw_int_fs = f.get_shape().as_list()
    raw_int_bs = b.get_shape().as_list()
    # extract patches from background with stride and rate
    kernel = 2*rate
    raw_w = tf.extract_image_patches(
        b, [1,kernel,kernel,1], [1,rate*stride,rate*stride,1], [1,1,1,1], padding='SAME')
    raw_w = tf.reshape(raw_w, [raw_int_bs[0], -1, kernel, kernel, raw_int_bs[3]])
    raw_w = tf.transpose(raw_w, [0, 2, 3, 4, 1])  # transpose to b*k*k*c*hw
    # downscaling foreground option: downscaling both foreground and
    # background for matching and use original background for reconstruction.
    f = resize(f, scale=1./rate, func=tf.image.resize_nearest_neighbor)
    b = resize(b, to_shape=[int(raw_int_bs[1]/rate), int(raw_int_bs[2]/rate)], func=tf.image.resize_nearest_neighbor)  # https://github.com/tensorflow/tensorflow/issues/11651
    if mask is not None:
        mask = resize(mask, scale=1./rate, func=tf.image.resize_nearest_neighbor)
    fs = tf.shape(f)
    int_fs = f.get_shape().as_list()
    f_groups = tf.split(f, int_fs[0], axis=0)
    # from t(H*W*C) to w(b*k*k*c*h*w)
    bs = tf.shape(b)
    int_bs = b.get_shape().as_list()
    w = tf.extract_image_patches(
        b, [1,ksize,ksize,1], [1,stride,stride,1], [1,1,1,1], padding='SAME')
    w = tf.reshape(w, [int_fs[0], -1, ksize, ksize, int_fs[3]])
    w = tf.transpose(w, [0, 2, 3, 4, 1])  # transpose to b*k*k*c*hw
    # process mask
    if mask is None:
        mask = tf.zeros([1, bs[1], bs[2], 1])
    m = tf.extract_image_patches(
        mask, [1,ksize,ksize,1], [1,stride,stride,1], [1,1,1,1], padding='SAME')
    m = tf.reshape(m, [1, -1, ksize, ksize, 1])
    m = tf.transpose(m, [0, 2, 3, 4, 1])  # transpose to b*k*k*c*hw
    m = m[0]
    mm = tf.cast(tf.equal(tf.reduce_mean(m, axis=[0,1,2], keep_dims=True), 0.), tf.float32)
    w_groups = tf.split(w, int_bs[0], axis=0)
    raw_w_groups = tf.split(raw_w, int_bs[0], axis=0)
    y = []
    offsets = []
    k = fuse_k
    scale = softmax_scale
    fuse_weight = tf.reshape(tf.eye(k), [k, k, 1, 1])
    for xi, wi, raw_wi in zip(f_groups, w_groups, raw_w_groups):
        # conv for compare
        wi = wi[0]
        wi_normed = wi / tf.maximum(tf.sqrt(tf.reduce_sum(tf.square(wi), axis=[0,1,2])), 1e-4)
        yi = tf.nn.conv2d(xi, wi_normed, strides=[1,1,1,1], padding="SAME")

        # conv implementation for fuse scores to encourage large patches
        if fuse:
            yi = tf.reshape(yi, [1, fs[1]*fs[2], bs[1]*bs[2], 1])
            yi = tf.nn.conv2d(yi, fuse_weight, strides=[1,1,1,1], padding='SAME')
            yi = tf.reshape(yi, [1, fs[1], fs[2], bs[1], bs[2]])
            yi = tf.transpose(yi, [0, 2, 1, 4, 3])
            yi = tf.reshape(yi, [1, fs[1]*fs[2], bs[1]*bs[2], 1])
            yi = tf.nn.conv2d(yi, fuse_weight, strides=[1,1,1,1], padding='SAME')
            yi = tf.reshape(yi, [1, fs[2], fs[1], bs[2], bs[1]])
            yi = tf.transpose(yi, [0, 2, 1, 4, 3])
        yi = tf.reshape(yi, [1, fs[1], fs[2], bs[1]*bs[2]])

        # softmax to match
        yi *=  mm  # mask
        yi = tf.nn.softmax(yi*scale, 3)
        yi *=  mm  # mask

        offset = tf.argmax(yi, axis=3, output_type=tf.int32)
        offset = tf.stack([offset // fs[2], offset % fs[2]], axis=-1)
        # deconv for patch pasting
        # 3.1 paste center
        wi_center = raw_wi[0]
        yi = tf.nn.conv2d_transpose(yi, wi_center, tf.concat([[1], raw_fs[1:]], axis=0), strides=[1,rate,rate,1]) / 4.
        y.append(yi)
        offsets.append(offset)
    y = tf.concat(y, axis=0)
    y.set_shape(raw_int_fs)
    offsets = tf.concat(offsets, axis=0)
    offsets.set_shape(int_bs[:3] + [2])
    # case1: visualize optical flow: minus current position
    h_add = tf.tile(tf.reshape(tf.range(bs[1]), [1, bs[1], 1, 1]), [bs[0], 1, bs[2], 1])
    w_add = tf.tile(tf.reshape(tf.range(bs[2]), [1, 1, bs[2], 1]), [bs[0], bs[1], 1, 1])
    offsets = offsets - tf.concat([h_add, w_add], axis=3)
    # to flow image
    flow = flow_to_image_tf(offsets)
    # # case2: visualize which pixels are attended
    # flow = highlight_flow_tf(offsets * tf.cast(mask, tf.int32))
    if rate != 1:
        flow = resize(flow, scale=rate, func=tf.image.resize_nearest_neighbor)
    return y, flow


def test_contextual_attention(args):
    """Test contextual attention layer with 3-channel image input
    (instead of n-channel feature).

    """
    import cv2
    import os
    # run on cpu
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    rate = 2
    stride = 1
    grid = rate*stride

    b = cv2.imread(args.imageA)
    b = cv2.resize(b, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
    h, w, _ = b.shape
    b = b[:h//grid*grid, :w//grid*grid, :]
    b = np.expand_dims(b, 0)
    logger.info('Size of imageA: {}'.format(b.shape))

    f = cv2.imread(args.imageB)
    h, w, _ = f.shape
    f = f[:h//grid*grid, :w//grid*grid, :]
    f = np.expand_dims(f, 0)
    logger.info('Size of imageB: {}'.format(f.shape))

    with tf.Session() as sess:
        bt = tf.constant(b, dtype=tf.float32)
        ft = tf.constant(f, dtype=tf.float32)

        yt, flow = contextual_attention(
            ft, bt, stride=stride, rate=rate,
            training=False, fuse=False)
        y = sess.run(yt)
        cv2.imwrite(args.imageOut, y[0])


def make_color_wheel():
    RY, YG, GC, CB, BM, MR = (15, 6, 4, 11, 13, 6)
    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros([ncols, 3])
    col = 0
    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255*np.arange(0, RY) / RY))
    col += RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.transpose(np.floor(255*np.arange(0, YG) / YG))
    colorwheel[col:col+YG, 1] = 255
    col += YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.transpose(np.floor(255*np.arange(0, GC) / GC))
    col += GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.transpose(np.floor(255*np.arange(0, CB) / CB))
    colorwheel[col:col+CB, 2] = 255
    col += CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.transpose(np.floor(255*np.arange(0, BM) / BM))
    col += + BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col+MR, 0] = 255
    return colorwheel


COLORWHEEL = make_color_wheel()


def compute_color(u,v):
    h, w = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0
    # colorwheel = COLORWHEEL
    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)
    rad = np.sqrt(u**2+v**2)
    a = np.arctan2(-v, -u) / np.pi
    fk = (a+1) / 2 * (ncols - 1) + 1
    k0 = np.floor(fk).astype(int)
    k1 = k0 + 1
    k1[k1 == ncols+1] = 1
    f = fk - k0
    for i in range(np.size(colorwheel,1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0-1] / 255
        col1 = tmp[k1-1] / 255
        col = (1-f) * col0 + f * col1
        idx = rad <= 1
        col[idx] = 1-rad[idx]*(1-col[idx])
        notidx = np.logical_not(idx)
        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col*(1-nanIdx)))
    return img



def flow_to_image(flow):
    """Transfer flow map to image.
    Part of code forked from flownet.
    """
    out = []
    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.
    maxrad = -1
    for i in range(flow.shape[0]):
        u = flow[i, :, :, 0]
        v = flow[i, :, :, 1]
        idxunknow = (abs(u) > 1e7) | (abs(v) > 1e7)
        u[idxunknow] = 0
        v[idxunknow] = 0
        maxu = max(maxu, np.max(u))
        minu = min(minu, np.min(u))
        maxv = max(maxv, np.max(v))
        minv = min(minv, np.min(v))
        rad = np.sqrt(u ** 2 + v ** 2)
        maxrad = max(maxrad, np.max(rad))
        u = u/(maxrad + np.finfo(float).eps)
        v = v/(maxrad + np.finfo(float).eps)
        img = compute_color(u, v)
        out.append(img)
    return np.float32(np.uint8(out))


def flow_to_image_tf(flow, name='flow_to_image'):
    """Tensorflow ops for computing flow to image.
    """
    with tf.variable_scope(name), tf.device('/cpu:0'):
        img = tf.py_func(flow_to_image, [flow], tf.float32, stateful=False)
        img.set_shape(flow.get_shape().as_list()[0:-1]+[3])
        img = img / 127.5 - 1.
        return img


def highlight_flow(flow):
    """Convert flow into middlebury color code image.
    """
    out = []
    s = flow.shape
    for i in range(flow.shape[0]):
        img = np.ones((s[1], s[2], 3)) * 144.
        u = flow[i, :, :, 0]
        v = flow[i, :, :, 1]
        for h in range(s[1]):
            for w in range(s[1]):
                ui = u[h,w]
                vi = v[h,w]
                img[ui, vi, :] = 255.
        out.append(img)
    return np.float32(np.uint8(out))


def highlight_flow_tf(flow, name='flow_to_image'):
    """Tensorflow ops for highlight flow.
    """
    with tf.variable_scope(name), tf.device('/cpu:0'):
        img = tf.py_func(highlight_flow, [flow], tf.float32, stateful=False)
        img.set_shape(flow.get_shape().as_list()[0:-1]+[3])
        img = img / 127.5 - 1.
        return img


def image2edge(image):
    """Convert image to edges.
    """
    out = []
    for i in range(image.shape[0]):
        img = cv2.Laplacian(image[i, :, :, :], cv2.CV_64F, ksize=3, scale=2)
        out.append(img)
    return np.float32(np.uint8(out))



logger = logging.getLogger()


class InpaintCAModel(Model):
    def __init__(self):
        Model.__init__(self, 'InpaintCAModel')

    def build_inpaint_net(self, x, mask, config=None, reuse=False,
                          training=True, padding='SAME', name='inpaint_net'):
        """Inpaint network.

        Args:
            x: incomplete image, [-1, 1]
            mask: mask region {0, 1}
        Returns:
            [-1, 1] as predicted image
        """
        xin = x
        offset_flow = None
        ones_x = tf.ones_like(x)[:, :, :, 0:1]
        x = tf.concat([x, ones_x, ones_x*mask], axis=3)

        # two stage network
        cnum = 32
        with tf.variable_scope(name, reuse=reuse), \
                arg_scope([gen_conv, gen_deconv],
                          training=training, padding=padding):
            # stage1
            x = gen_conv(x, cnum, 5, 1, name='conv1')
            x = gen_conv(x, 2*cnum, 3, 2, name='conv2_downsample')
            x = gen_conv(x, 2*cnum, 3, 1, name='conv3')
            x = gen_conv(x, 4*cnum, 3, 2, name='conv4_downsample')
            x = gen_conv(x, 4*cnum, 3, 1, name='conv5')
            x = gen_conv(x, 4*cnum, 3, 1, name='conv6')
            mask_s = resize_mask_like(mask, x)
            x = gen_conv(x, 4*cnum, 3, rate=2, name='conv7_atrous')
            x = gen_conv(x, 4*cnum, 3, rate=4, name='conv8_atrous')
            x = gen_conv(x, 4*cnum, 3, rate=8, name='conv9_atrous')
            x = gen_conv(x, 4*cnum, 3, rate=16, name='conv10_atrous')
            x = gen_conv(x, 4*cnum, 3, 1, name='conv11')
            x = gen_conv(x, 4*cnum, 3, 1, name='conv12')
            x = gen_deconv(x, 2*cnum, name='conv13_upsample')
            x = gen_conv(x, 2*cnum, 3, 1, name='conv14')
            x = gen_deconv(x, cnum, name='conv15_upsample')
            x = gen_conv(x, cnum//2, 3, 1, name='conv16')
            x = gen_conv(x, 3, 3, 1, activation=None, name='conv17')
            x = tf.clip_by_value(x, -1., 1.)
            x_stage1 = x
            # return x_stage1, None, None

            # stage2, paste result as input
            # x = tf.stop_gradient(x)
            x = x*mask + xin*(1.-mask)
            x.set_shape(xin.get_shape().as_list())
            # conv branch
            xnow = tf.concat([x, ones_x, ones_x*mask], axis=3)
            x = gen_conv(xnow, cnum, 5, 1, name='xconv1')
            x = gen_conv(x, cnum, 3, 2, name='xconv2_downsample')
            x = gen_conv(x, 2*cnum, 3, 1, name='xconv3')
            x = gen_conv(x, 2*cnum, 3, 2, name='xconv4_downsample')
            x = gen_conv(x, 4*cnum, 3, 1, name='xconv5')
            x = gen_conv(x, 4*cnum, 3, 1, name='xconv6')
            x = gen_conv(x, 4*cnum, 3, rate=2, name='xconv7_atrous')
            x = gen_conv(x, 4*cnum, 3, rate=4, name='xconv8_atrous')
            x = gen_conv(x, 4*cnum, 3, rate=8, name='xconv9_atrous')
            x = gen_conv(x, 4*cnum, 3, rate=16, name='xconv10_atrous')
            x_hallu = x
            # attention branch
            x = gen_conv(xnow, cnum, 5, 1, name='pmconv1')
            x = gen_conv(x, cnum, 3, 2, name='pmconv2_downsample')
            x = gen_conv(x, 2*cnum, 3, 1, name='pmconv3')
            x = gen_conv(x, 4*cnum, 3, 2, name='pmconv4_downsample')
            x = gen_conv(x, 4*cnum, 3, 1, name='pmconv5')
            x = gen_conv(x, 4*cnum, 3, 1, name='pmconv6',
                         activation=tf.nn.relu)
            x, offset_flow = contextual_attention(x, x, mask_s, 3, 1, rate=2)
            x = gen_conv(x, 4*cnum, 3, 1, name='pmconv9')
            x = gen_conv(x, 4*cnum, 3, 1, name='pmconv10')
            pm = x
            x = tf.concat([x_hallu, pm], axis=3)

            x = gen_conv(x, 4*cnum, 3, 1, name='allconv11')
            x = gen_conv(x, 4*cnum, 3, 1, name='allconv12')
            x = gen_deconv(x, 2*cnum, name='allconv13_upsample')
            x = gen_conv(x, 2*cnum, 3, 1, name='allconv14')
            x = gen_deconv(x, cnum, name='allconv15_upsample')
            x = gen_conv(x, cnum//2, 3, 1, name='allconv16')
            x = gen_conv(x, 3, 3, 1, activation=None, name='allconv17')
            x_stage2 = tf.clip_by_value(x, -1., 1.)
        return x_stage1, x_stage2, offset_flow

    def build_wgan_local_discriminator(self, x, reuse=False, training=True):
        with tf.variable_scope('discriminator_local', reuse=reuse):
            cnum = 64
            x = dis_conv(x, cnum, name='conv1', training=training)
            x = dis_conv(x, cnum*2, name='conv2', training=training)
            x = dis_conv(x, cnum*4, name='conv3', training=training)
            x = dis_conv(x, cnum*8, name='conv4', training=training)
            x = flatten(x, name='flatten')
            return x

    def build_wgan_global_discriminator(self, x, reuse=False, training=True):
        with tf.variable_scope('discriminator_global', reuse=reuse):
            cnum = 64
            x = dis_conv(x, cnum, name='conv1', training=training)
            x = dis_conv(x, cnum*2, name='conv2', training=training)
            x = dis_conv(x, cnum*4, name='conv3', training=training)
            x = dis_conv(x, cnum*4, name='conv4', training=training)
            x = flatten(x, name='flatten')
            return x

    def build_wgan_discriminator(self, batch_local, batch_global,
                                 reuse=False, training=True):
        with tf.variable_scope('discriminator', reuse=reuse):
            dlocal = self.build_wgan_local_discriminator(
                batch_local, reuse=reuse, training=training)
            dglobal = self.build_wgan_global_discriminator(
                batch_global, reuse=reuse, training=training)
            dout_local = tf.layers.dense(dlocal, 1, name='dout_local_fc')
            dout_global = tf.layers.dense(dglobal, 1, name='dout_global_fc')
            return dout_local, dout_global

    def build_graph_with_losses(self, batch_data, config, training=True,
                                summary=False, reuse=False):
        batch_pos = batch_data / 127.5 - 1.
        # generate mask, 1 represents masked point
        bbox = random_bbox(config)
        mask = bbox2mask(bbox, config, name='mask_c')
        batch_incomplete = batch_pos*(1.-mask)
        x1, x2, offset_flow = self.build_inpaint_net(
            batch_incomplete, mask, config, reuse=reuse, training=training,
            padding=config.PADDING)
        if config.PRETRAIN_COARSE_NETWORK:
            batch_predicted = x1
            logger.info('Set batch_predicted to x1.')
        else:
            batch_predicted = x2
            logger.info('Set batch_predicted to x2.')
        losses = {}
        # apply mask and complete image
        batch_complete = batch_predicted*mask + batch_incomplete*(1.-mask)
        # local patches
        local_patch_batch_pos = local_patch(batch_pos, bbox)
        local_patch_batch_predicted = local_patch(batch_predicted, bbox)
        local_patch_x1 = local_patch(x1, bbox)
        local_patch_x2 = local_patch(x2, bbox)
        local_patch_batch_complete = local_patch(batch_complete, bbox)
        local_patch_mask = local_patch(mask, bbox)
        l1_alpha = config.COARSE_L1_ALPHA
        losses['l1_loss'] = l1_alpha * tf.reduce_mean(tf.abs(local_patch_batch_pos - local_patch_x1)*spatial_discounting_mask(config))
        if not config.PRETRAIN_COARSE_NETWORK:
            losses['l1_loss'] += tf.reduce_mean(tf.abs(local_patch_batch_pos - local_patch_x2)*spatial_discounting_mask(config))
        losses['ae_loss'] = l1_alpha * tf.reduce_mean(tf.abs(batch_pos - x1) * (1.-mask))
        if not config.PRETRAIN_COARSE_NETWORK:
            losses['ae_loss'] += tf.reduce_mean(tf.abs(batch_pos - x2) * (1.-mask))
        losses['ae_loss'] /= tf.reduce_mean(1.-mask)
        if summary:
            scalar_summary('losses/l1_loss', losses['l1_loss'])
            scalar_summary('losses/ae_loss', losses['ae_loss'])
            viz_img = [batch_pos, batch_incomplete, batch_complete]
            if offset_flow is not None:
                viz_img.append(
                    resize(offset_flow, scale=4,
                           func=tf.image.resize_nearest_neighbor))
            images_summary(
                tf.concat(viz_img, axis=2),
                'raw_incomplete_predicted_complete', config.VIZ_MAX_OUT)

        # gan
        batch_pos_neg = tf.concat([batch_pos, batch_complete], axis=0)
        # local deterministic patch
        local_patch_batch_pos_neg = tf.concat([local_patch_batch_pos, local_patch_batch_complete], 0)
        if config.GAN_WITH_MASK:
            batch_pos_neg = tf.concat([batch_pos_neg, tf.tile(mask, [config.BATCH_SIZE*2, 1, 1, 1])], axis=3)
        # wgan with gradient penalty
        if config.GAN == 'wgan_gp':
            # seperate gan
            pos_neg_local, pos_neg_global = self.build_wgan_discriminator(local_patch_batch_pos_neg, batch_pos_neg, training=training, reuse=reuse)
            pos_local, neg_local = tf.split(pos_neg_local, 2)
            pos_global, neg_global = tf.split(pos_neg_global, 2)
            # wgan loss
            g_loss_local, d_loss_local = gan_wgan_loss(pos_local, neg_local, name='gan/local_gan')
            g_loss_global, d_loss_global = gan_wgan_loss(pos_global, neg_global, name='gan/global_gan')
            losses['g_loss'] = config.GLOBAL_WGAN_LOSS_ALPHA * g_loss_global + g_loss_local
            losses['d_loss'] = d_loss_global + d_loss_local
            # gp
            interpolates_local = random_interpolates(local_patch_batch_pos, local_patch_batch_complete)
            interpolates_global = random_interpolates(batch_pos, batch_complete)
            dout_local, dout_global = self.build_wgan_discriminator(
                interpolates_local, interpolates_global, reuse=True)
            # apply penalty
            penalty_local = gradients_penalty(interpolates_local, dout_local, mask=local_patch_mask)
            penalty_global = gradients_penalty(interpolates_global, dout_global, mask=mask)
            losses['gp_loss'] = config.WGAN_GP_LAMBDA * (penalty_local + penalty_global)
            losses['d_loss'] = losses['d_loss'] + losses['gp_loss']
            if summary and not config.PRETRAIN_COARSE_NETWORK:
                gradients_summary(g_loss_local, batch_predicted, name='g_loss_local')
                gradients_summary(g_loss_global, batch_predicted, name='g_loss_global')
                scalar_summary('convergence/d_loss', losses['d_loss'])
                scalar_summary('convergence/local_d_loss', d_loss_local)
                scalar_summary('convergence/global_d_loss', d_loss_global)
                scalar_summary('gan_wgan_loss/gp_loss', losses['gp_loss'])
                scalar_summary('gan_wgan_loss/gp_penalty_local', penalty_local)
                scalar_summary('gan_wgan_loss/gp_penalty_global', penalty_global)

        if summary and not config.PRETRAIN_COARSE_NETWORK:
            # summary the magnitude of gradients from different losses w.r.t. predicted image
            gradients_summary(losses['g_loss'], batch_predicted, name='g_loss')
            gradients_summary(losses['g_loss'], x1, name='g_loss_to_x1')
            gradients_summary(losses['g_loss'], x2, name='g_loss_to_x2')
            gradients_summary(losses['l1_loss'], x1, name='l1_loss_to_x1')
            gradients_summary(losses['l1_loss'], x2, name='l1_loss_to_x2')
            gradients_summary(losses['ae_loss'], x1, name='ae_loss_to_x1')
            gradients_summary(losses['ae_loss'], x2, name='ae_loss_to_x2')
        if config.PRETRAIN_COARSE_NETWORK:
            losses['g_loss'] = 0
        else:
            losses['g_loss'] = config.GAN_LOSS_ALPHA * losses['g_loss']
        losses['g_loss'] += config.L1_LOSS_ALPHA * losses['l1_loss']
        logger.info('Set L1_LOSS_ALPHA to %f' % config.L1_LOSS_ALPHA)
        logger.info('Set GAN_LOSS_ALPHA to %f' % config.GAN_LOSS_ALPHA)
        if config.AE_LOSS:
            losses['g_loss'] += config.AE_LOSS_ALPHA * losses['ae_loss']
            logger.info('Set AE_LOSS_ALPHA to %f' % config.AE_LOSS_ALPHA)
        g_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, 'inpaint_net')
        d_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
        return g_vars, d_vars, losses

    def build_infer_graph(self, batch_data, config, bbox=None, name='val'):
        """
        """
        config.MAX_DELTA_HEIGHT = 0
        config.MAX_DELTA_WIDTH = 0
        if bbox is None:
            bbox = random_bbox(config)
        mask = bbox2mask(bbox, config, name=name+'mask_c')
        batch_pos = batch_data / 127.5 - 1.
        edges = None
        batch_incomplete = batch_pos*(1.-mask)
        # inpaint
        x1, x2, offset_flow = self.build_inpaint_net(
            batch_incomplete, mask, config, reuse=True,
            training=False, padding=config.PADDING)
        if config.PRETRAIN_COARSE_NETWORK:
            batch_predicted = x1
            logger.info('Set batch_predicted to x1.')
        else:
            batch_predicted = x2
            logger.info('Set batch_predicted to x2.')
        # apply mask and reconstruct
        batch_complete = batch_predicted*mask + batch_incomplete*(1.-mask)
        # global image visualization
        viz_img = [batch_pos, batch_incomplete, batch_complete]
        if offset_flow is not None:
            viz_img.append(
                resize(offset_flow, scale=4,
                       func=tf.image.resize_nearest_neighbor))
        images_summary(
            tf.concat(viz_img, axis=2),
            name+'_raw_incomplete_complete', config.VIZ_MAX_OUT)
        return batch_complete

    def build_static_infer_graph(self, batch_data, config, name):
        """
        """
        # generate mask, 1 represents masked point
        bbox = (tf.constant(config.HEIGHT//2), tf.constant(config.WIDTH//2),
                tf.constant(config.HEIGHT), tf.constant(config.WIDTH))
        return self.build_infer_graph(batch_data, config, bbox, name)


    def build_server_graph(self, batch_data, reuse=False, is_training=False):
        """
        """
        # generate mask, 1 represents masked point
        batch_raw, masks_raw = tf.split(batch_data, 2, axis=2)
        masks = tf.cast(masks_raw[0:1, :, :, 0:1] > 127.5, tf.float32)

        batch_pos = batch_raw / 127.5 - 1.
        batch_incomplete = batch_pos * (1. - masks)
        # inpaint
        x1, x2, flow = self.build_inpaint_net(
            batch_incomplete, masks, reuse=reuse, training=is_training,
            config=None)
        batch_predict = x2
        # apply mask and reconstruct
        batch_complete = batch_predict*masks + batch_incomplete*(1-masks)
        return batch_complete

def channelData(layer):
    region = layer.get_pixel_rgn(0, 0, layer.width,layer.height)
    pixChars = region[:,:] # Take whole layer
    bpp = region.bpp
    return np.frombuffer(pixChars,dtype=np.uint8).reshape(layer.height, layer.width, bpp)
    #return np.frombuffer(pixChars,dtype=np.uint8).reshape(len(pixChars)/bpp,bpp)

def createResultLayer(image, name, result):
    pixels = np.uint8(result)
    print(pixels.shape)
    rlBytes = pixels.tobytes();
    rl = gimp.Layer(image,name,image.width,image.height,image.active_layer.type,100,NORMAL_MODE)
    region = rl.get_pixel_rgn(0, 0, rl.width,rl.height,True)
    print(len(region[:,:]))
    print(len(rlBytes))
    region[:,:] = rlBytes
    image.add_layer(rl,0)
    rl.flush()
    rl.update(0,0,rl.width,rl.height)
    #pdb.gimp_edit_clear(rl)
    return rl

def fillLayer(layer, result):
    pixels = np.uint8(result)
    print(pixels.shape)
    print(layer.width, layer.height)
    rlBytes = pixels.tobytes();
    #rl = gimp.Layer(image,name,image.width,image.height,image.active_layer.type,100,NORMAL_MODE)
    region = layer.get_pixel_rgn(0, 0, layer.width, layer.height,True)
    region[:,:] = rlBytes
    #image.add_layer(rl,0)
    #gimp.displays_flush()

def python_generative_infill(timg, tdrawable, bx=9):
    width = tdrawable.width
    height = tdrawable.height

    ng.get_gpus(1, dedicated=False)
    model = InpaintCAModel()
    checkpoint_dir = "model_logs/places2"
    imgdata = channelData(tdrawable)
    print(imgdata.shape)
    print(imgdata)
    image = imgdata[:,:,:3][:,:,::-1]
    #Mask fully white areas
    mask = ((image == np.array([255,255,255]))*255)
    print(np.sum(mask))

    #image = cv2.imread(args.image)
    #mask = cv2.imread(args.mask)
    #assert image.shape == mask.shape

    #h, w, _ = image.shape
    h = height
    w = width
    grid = 8

    print(grid, w, h)
    print(image.shape)
    #image = image[:h//grid*grid, :w//grid*grid, :]
    print(image)
    print(image.shape)
    print(mask.shape)
    #mask = mask[:h//grid*grid, :w//grid*grid, :]
    print(mask.shape)

    image = np.expand_dims(image, 0)
    mask = np.expand_dims(mask, 0)
    input_image = np.concatenate([image, mask], axis=2)
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    with tf.Session(config=sess_config) as sess:
        input_image = tf.constant(input_image, dtype=tf.float32)
        output = model.build_server_graph(input_image)
        output = (output + 1.) * 127.5
        output = tf.reverse(output, [-1])
        output = tf.saturate_cast(output, tf.uint8)
        # load pretrained model
        vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        assign_ops = []
        for var in vars_list:
            vname = var.name
            from_name = vname
            var_value = tf.contrib.framework.load_variable(checkpoint_dir, from_name)
            assign_ops.append(tf.assign(var, var_value))
        sess.run(assign_ops)
        print('Model loaded.')
        result = sess.run(output)
        print(result.shape)
        #cv2.imwrite(args.output, result[0][:, :, ::-1])
        #rgb = np.uint8(result[0][:, :, ::-1])
        rgb = np.uint8(result[0][:, :, :])
        pixels = np.concatenate((rgb, 255*np.ones((rgb.shape[0], rgb.shape[1], 1))), axis=2)
        rl = createResultLayer(timg, 'result', pixels)
#        rl = createResultLayer(timg, 'result', result[0][:, :, ::-1])
        if 0:
		rgb = np.uint8(result[0][:, :, ::-1])
		print(rgb.shape)
		pixels = np.concatenate((rgb, 255*np.ones((rgb.shape[0], rgb.shape[1], 1))), axis=2)
		print(pixels.shape)
		#print(pixels.shape)
		#rlBytes = rgb.tobytes()
		rlBytes = pixels.tobytes();
		rl = tdrawable
		#rl = gimp.Layer(img,name,image.width,image.height,image.active_layer.type,100,NORMAL_MODE)
		region = rl.get_pixel_rgn(0, 0, rl.width,rl.height,True)
		print(len(region[:,:]))
		print(len(rlBytes))
		region[:,:] = rlBytes
		#image.add_layer(rl,0)
		rl.flush()
		rl.update(0,0,rl.width,rl.height)
    
    #img.flatten()
    #img.remove_layer(layer_one)

    #bump_layer = img.active_layer

    #pdb.plug_in_c_astretch(img, bump_layer)
    #pdb.plug_in_noisify(img, bump_layer, 0, 0.2, 0.2, 0.2, 0.2)
    #pdb.plug_in_bump_map(img, tdrawable, bump_layer, azimuth,
    #                     elevation, depth, 0, 0, 0, 0, True, False, 0)
    pdb.gimp_edit_clear(tdrawable)
    #img.flush()
    #pdb.gimp_undo_push_group_end(img)
    #gimp.delete(img)
    gimp.displays_flush()

register(
        "python_generative_inpainting",
        "Generate pixels for transparent sections of an image",
        "Generate pixels for transparent sections of an image",
        "Lasse Hyyyrynen",
        "Lasse Hyyrynen",
        "2018",
        "<Image>/Filters/Enhance/_Generate_pixels...",
        "RGB*, GRAY*",
        [
                (PF_INT, "x_blur", "Irrelevant argument. You can ignore this now.", 9)
        ],
        [],
        python_generative_infill)

main()


