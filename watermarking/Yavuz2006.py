"""
Yavuz, E., & Telatar, Z. (2007). 
Improved SVD-DWT based digital image watermarking against watermark ambiguity. 
Proceedings of the 2007 ACM Symposium on Applied Computing - SAC ’07, 1051. 
https://doi.org/10.1145/1244002.1244232
"""

from numpy import empty, empty_like
from scipy.fft import dctn, idctn

from Chandra2002 import Watermarker as chandraWatermarker


class Watermarker:
    IMAGE_TO_WATERMARK_RATIO = 8

    def __init__(self, scale_factor=2 ** (-4)) -> None:
        self.sf = scale_factor

    def add_watermark(self, host, watermark):
        R = self.IMAGE_TO_WATERMARK_RATIO

        watermarked = empty_like(host)

        for i in range(0, host.shape[0], R):
            for j in range(0, host.shape[1], R):
                watermarked[i : i + R, j : j + R] = dctn(host[i : i + R, j : j + R])

                watermarked[i, j + 1] += self.sf * watermark[i // R, j // R]
                watermarked[i, j + 2] += self.sf * watermark[i // R, j // R]

        self.svd_watermarker = chandraWatermarker(self.sf)
        watermarked[::R, ::R] = self.svd_watermarker.add_watermark(
            watermarked[::R, ::R], watermark
        )

        for i in range(0, host.shape[0], R):
            for j in range(0, host.shape[1], R):
                watermarked[i : i + R, j : j + R] = idctn(
                    watermarked[i : i + R, j : j + R]
                )

        return watermarked

    def check_watermark(self, watermarked_image):
        pass

    def extract_watermark(self, watermarked_image):
        R = self.IMAGE_TO_WATERMARK_RATIO

        watermarked_ = empty(
            shape=(
                watermarked_image.shape[0] // R,
                watermarked_image.shape[1] // R,
            ),
            dtype=watermarked_image.dtype,
        )

        for i in range(0, watermarked_image.shape[0], R):
            for j in range(0, watermarked_image.shape[1], R):
                watermarked_[i // R, j // R] = dctn(
                    watermarked_image[i : i + R, j : j + R]
                )[0, 0]

        return self.svd_watermarker.extract_watermark(watermarked_)
