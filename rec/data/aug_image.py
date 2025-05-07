# @title Augment image
import tensorflow as tf
import numpy as np
import random
import math
import re
import cv2
from functools import partial
import PIL
from PIL import Image, ImageChops, ImageEnhance, ImageFilter, ImageOps

_PIL_VER = tuple([int(x) for x in PIL.__version__.split('.')[:2]])

class ImageAugmentation:
    def __init__(self):
        self._LEVEL_DENOM = 10.
        self._FILL = [128, 128, 128]
        self.ops = {
            'AutoContrast': self.auto_contrast,
            'Equalize': self.equalize,
            'Invert': self.invert,
            'Rotate': self.rotate,
            'Posterize': self.posterize,
            'PosterizeIncreasing': self.posterize,
            'Solarize': self.solarize,
            'SolarizeIncreasing': self.solarize,
            'SolarizeAdd': self.solarize_add,
            'Color': self.color,
            'ColorIncreasing': self.color,
            'Contrast': self.contrast,
            'ContrastIncreasing': self.contrast,
            'Brightness': self.brightness,
            'BrightnessIncreasing': self.brightness,
            'Sharpness': self.sharpness,
            'SharpnessIncreasing': self.sharpness,
            'ShearX': self.shear_x,
            'ShearY': self.shear_y,
            'TranslateX': self.translate_x_abs,
            'TranslateY': self.translate_y_abs,
            'TranslateXRel': self.translate_x_rel,
            'TranslateYRel': self.translate_y_rel,
            'Desaturate': self.desaturate,
            'GaussianBlur': self.gaussian_blur,
            'GaussianBlurRand': self.gaussian_blur_rand,
        }

        self.level_map = {
            'AutoContrast': None,
            'Equalize': None,
            'Invert': None,
            'Rotate': self._rotate_level_to_arg,
            'Posterize': self._posterize_level_to_arg,
            'PosterizeIncreasing': self._posterize_increasing_level_to_arg,
            'Solarize': self._solarize_level_to_arg,
            'SolarizeIncreasing': self._solarize_increasing_level_to_arg,
            'SolarizeAdd': self._solarize_add_level_to_arg,
            'Color': self._enhance_level_to_arg,
            'ColorIncreasing': self._enhance_increasing_level_to_arg,
            'Contrast': self._enhance_level_to_arg,
            'ContrastIncreasing': self._enhance_increasing_level_to_arg,
            'Brightness': self._enhance_level_to_arg,
            'BrightnessIncreasing': self._enhance_increasing_level_to_arg,
            'Sharpness': self._enhance_level_to_arg,
            'SharpnessIncreasing': self._enhance_increasing_level_to_arg,
            'ShearX': self._shear_level_to_arg,
            'ShearY': self._shear_level_to_arg,
            'TranslateX': self._translate_abs_level_to_arg,
            'TranslateY': self._translate_abs_level_to_arg,
            'TranslateXRel': self._translate_rel_level_to_arg,
            'TranslateYRel': self._translate_rel_level_to_arg,
            'Desaturate': partial(self._minmax_level_to_arg, min_val=0.5, max_val=1.0),
            'GaussianBlur': partial(self._minmax_level_to_arg, min_val=0.1, max_val=2.0),
            'GaussianBlurRand': self._minmax_level_to_arg,
        }
    # Level function
    def _randomly_negate(self, v):
        return -v if random.random() > 0.5 else v

    def _rotate_level_to_arg(self, level, hparams):
        degree = hparams.get('rotate_deg', 30.)
        level = (level / self._LEVEL_DENOM) * degree
        level = self._randomly_negate(level)
        return (level,)

    def _enhance_level_to_arg(self, level, hparams):
        return ((level / self._LEVEL_DENOM) * 1.8 + 0.1,)

    def _enhance_increasing_level_to_arg(self, level, hparams):
        level = (level / self._LEVEL_DENOM) * .9
        level = max(0.1, 1.0 + self._randomly_negate(level))
        return (level,)

    def _minmax_level_to_arg(self, level, hparams, min_val=0., max_val=1.0, clamp=True):
        level = (level / self._LEVEL_DENOM)
        level = min_val + (max_val - min_val) * level
        if clamp:
            level = max(min_val, min(max_val, level))
        return (level,)

    def _shear_level_to_arg(self, level, hparams):
        shear = hparams.get('shear_pct', 0.2)
        level = (level / self._LEVEL_DENOM) * shear
        level = self._randomly_negate(level)
        return (level,)

    def _translate_abs_level_to_arg(self, level, hparams):
        translate_const = hparams.get('translate_const', 250)
        level = (level / self._LEVEL_DENOM) * float(translate_const)
        level = self._randomly_negate(level)
        return (level,)

    def _translate_rel_level_to_arg(self, level, hparams):
        translate_pct = hparams.get('translate_pct', 0.45)
        level = (level / self._LEVEL_DENOM) * translate_pct
        level = self._randomly_negate(level)
        return (level,)

    def _posterize_level_to_arg(self, level, hparams):
        return (int((level / self._LEVEL_DENOM) * 4),)

    def _posterize_increasing_level_to_arg(self, level, hparams):
        return (4 - self._posterize_level_to_arg(level, hparams)[0],)

    def _solarize_level_to_arg(self, level, hparams):
        return (min(256, int((level / self._LEVEL_DENOM) * 256)),)

    def _solarize_increasing_level_to_arg(self, level, hparams):
        return (256 - self._solarize_level_to_arg(level, hparams)[0],)

    def _solarize_add_level_to_arg(self, level, hparams):
        return (min(128, int((level / self._LEVEL_DENOM) * 110)),)

    # Transformation functions
    def auto_contrast(self, img, *args, **kwargs):
        return ImageOps.autocontrast(img)

    def equalize(self, img, *args, **kwargs):
        return ImageOps.equalize(img)

    def invert(self, img, *args, **kwargs):
        return ImageOps.invert(img)

    def rotate(self, img, degrees, **kwargs):
        if _PIL_VER >= (5, 2):
            return img.rotate(degrees, **kwargs)
        if _PIL_VER >= (5, 0):
            w, h = img.size
            post_trans = (0, 0)
            rotn_center = (w / 2.0, h / 2.0)
            angle = -math.radians(degrees)
            matrix = [
                round(math.cos(angle), 15),
                round(math.sin(angle), 15),
                0.0,
                round(-math.sin(angle), 15),
                round(math.cos(angle), 15),
                0.0,
            ]

            def transform(x, y, matrix):
                (a, b, c, d, e, f) = matrix
                return a * x + b * y + c, d * x + e * y + f

            matrix[2], matrix[5] = transform(-rotn_center[0] - post_trans[0],
                                            -rotn_center[1] - post_trans[1],
                                            matrix)
            matrix[2] += rotn_center[0]
            matrix[5] += rotn_center[1]
            return img.transform(img.size, Image.AFFINE, matrix, **kwargs)
        return img.rotate(degrees, resample=kwargs['resample'])

    def posterize(self, image, bits_to_keep, **kwargs):
        if bits_to_keep >= 8:
            return image
        return ImageOps.posterize(image, bits_to_keep)

    def solarize(self, image, thresh, **kwargs):
        return ImageOps.solarize(image, thresh)

    def solarize_add(self, img, add, thresh=128, **kwargs):
        lut = []
        for i in range(256):
            if i < thresh:
                lut.append(min(255, i + add))
            else:
                lut.append(i)

        if img.mode in ('L', 'RGB'):
            if img.mode == 'RGB' and len(lut) == 256:
                lut = lut + lut + lut
            return img.point(lut)

        return img

    def color(self, img, factor, **kwargs):
        return ImageEnhance.Color(img).enhance(factor)

    def contrast(self, img, factor, **kwargs):
        return ImageEnhance.Contrast(img).enhance(factor)

    def brightness(self, img, factor, **kwargs):
        return ImageEnhance.Brightness(img).enhance(factor)

    def sharpness(self, img, factor, **kwargs):
        return ImageEnhance.Sharpness(img).enhance(factor)

    def shear_x(self, img, factor, **kwargs):
        return img.transform(img.size, Image.AFFINE, (1, factor, 0, 0, 1, 0),
                         **kwargs)

    def shear_y(self, img, factor, **kwargs):
        return img.transform(img.size, Image.AFFINE, (1, 0, 0, factor, 1, 0),
                         **kwargs)

    def translate_x_abs(self, img, pixels, **kwargs):
        return img.transform(img.size, Image.AFFINE, (1, 0, pixels, 0, 1, 0),
                         **kwargs)

    def translate_y_abs(self, img, pixels, **kwargs):
        return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, pixels),
                         **kwargs)

    def translate_x_rel(self, img, pct, **kwargs):
        pixels = pct * img.size[0]
        return img.transform(img.size, Image.AFFINE, (1, 0, pixels, 0, 1, 0),
                            **kwargs)


    def translate_y_rel(self, img, pct, **kwargs):
        pixels = pct * img.size[1]
        return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, pixels),
                            **kwargs)

    def desaturate(self, img, factor, **_):
        factor = min(1., max(0., 1. - factor))
        return ImageEnhance.Color(img).enhance(factor)

    def gaussian_blur(self, img, factor, **__):
        img = img.filter(ImageFilter.GaussianBlur(radius=factor))
        return img


    def gaussian_blur_rand(self, img, factor, **__):
        radius_min = 0.1
        radius_max = 2.0
        img = img.filter(
            ImageFilter.GaussianBlur(radius=random.uniform(radius_min, radius_max *
                                                          factor)))
        return img

    def apply_transform(self, image, name, magnitude, hparams, prob=0.5):
        if prob < 1.0 and random.random() > prob:
            return image

        # Asegurarse de que la imagen es un objeto PIL Image
        if isinstance(image, np.ndarray):
            if image.shape[2] == 3:
                image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                image = Image.fromarray(image)

        op = self.ops[name]
        level_fn = self.level_map[name]

        if level_fn is not None:
            args = level_fn(magnitude, hparams)
        else:
            args = tuple()

        return op(image, *args)