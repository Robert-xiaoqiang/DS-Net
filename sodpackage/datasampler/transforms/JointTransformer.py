import random

import numpy as np
import torch
from PIL import Image, ImageFilter
from torchvision.transforms import transforms


class JointCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, *args):
        for t in self.transforms:
            args = t(*args)
        return args


class JointRandomHorizontallyFlip(object):
    def __call__(self, *args):
        ret = list(args)
        if random.random() < 0.5:
            for i, arg in enumerate(args):
                ret[i] = arg.transpose(Image.FLIP_LEFT_RIGHT)
        return tuple(ret)

class JointResize(object):
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        elif isinstance(size, (tuple, list)):
            self.size = size
        else:
            raise RuntimeError("size参数请设置为int或者tuple, list")
    
    def __call__(self, *args):
        return tuple([ arg.resize(self.size) for arg in args ])

class JointRandomRotate(object):
    def __init__(self, degree):
        self.degree = degree
    
    def __call__(self, *args):
        rotate_degree = random.random() * 2 * self.degree - self.degree
        return tuple([ x.rotate(rotate_degree, Image.BILINEAR) for x in args ])
