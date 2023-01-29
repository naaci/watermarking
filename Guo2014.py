"""
Guo, J.-M. M., & Prasetyo, H. (2014). 
False-positive-free SVD-based image watermarking. 
Journal of Visual Communication and Image Representation, 25(5), 1149–1163. 
https://doi.org/10.1016/j.jvcir.2014.03.012
"""

from numpy import empty, empty_like
from scipy.linalg import diagsvd, svd
# from numpy.linalg import svd


class Watermarker:
    IMAGE_TO_WATERMARK_RATIO = 4

    def __init__(self, scale_factor=.01) -> None:
        self.sf = scale_factor

    def add_watermark(self, host_image, watermark_image):
        R = host_image.shape[0] // watermark_image.shape[0]
        # R = host_image.shape[1] // watermark_image.shape[1]

        U_w, s_w, self.Vh_w = svd(watermark_image, full_matrices=True)
        P_w = U_w @ diagsvd(s_w, *watermark_image.shape)

        watermarked = empty_like(host_image)
        self.s = empty_like(watermark_image)
        for i in range(0, watermark_image.shape[0]):
            for j in range(0, watermark_image.shape[1]):
                U, s, Vh = svd(host_image[i * R:(i + 1) * R,
                                          j * R:(j + 1) * R],
                               full_matrices=False)
                self.s[i, j] = s[0]
                s[0] += self.sf * P_w[i, j]
                watermarked[i * R:(i + 1) * R,
                            j * R:(j + 1) * R] = (U * s) @ Vh

        return watermarked

    def extract_watermark(self, watermarked_image):
        R = self.IMAGE_TO_WATERMARK_RATIO
        P_w = empty(shape=(watermarked_image.shape[0] // R,
                           watermarked_image.shape[1] // R))
        for i in range(0, P_w.shape[0]):
            for j in range(0, P_w.shape[1]):
                s = svd(watermarked_image[i * R:(i + 1) * R,
                                          j * R:(j + 1) * R],
                        compute_uv=False)
                P_w[i, j] = (s[0] - self.s[i, j]) / self.sf
        return P_w @ self.Vh_w

