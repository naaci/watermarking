"""
Chang, C. C., Tsai, P., & Lin, C. C. (2005). 
SVD-based digital image watermarking scheme. 
Pattern Recognition Letters, 26(10), 1577–1586. 
https://doi.org/10.1016/j.patrec.2005.01.004
"""

from numpy import abs, angle, empty, empty_like, exp, sign
from numpy.linalg import svd


class Watermarker:
    IMAGE_TO_WATERMARK_RATIO = 4

    def __init__(self, scale_factor=.012) -> None:
        self.sf = scale_factor

    def add_watermark(self, host_image, watermark_image):
        self.R1 = host_image.shape[0] // watermark_image.shape[0]
        self.R2 = host_image.shape[1] // watermark_image.shape[1]

        watermarked = empty_like(host_image)
        for i in range(0, watermark_image.shape[0]):
            for j in range(0, watermark_image.shape[1]):
                U, s, Vh = svd(host_image[i * self.R1:(i + 1) * self.R1,
                                          j * self.R2:(j + 1) * self.R2],
                               full_matrices=False)
                # complexity = s[s != 0].size / s.size

                m = (abs(U[1, 0]) + abs(U[2, 0])) / 2
                d = abs(U[1, 0]) - abs(U[2, 0])

                if watermark_image[i, j] < .5:
                    if d < self.sf:
                        U[1, 0] = (m + self.sf / 2)*exp(angle(U[1, 0])*1j)
                        U[2, 0] = (m - self.sf / 2)*exp(angle(U[2, 0])*1j)

                else:
                    if -d < self.sf:
                        U[1, 0] = (m - self.sf / 2)*exp(angle(U[1, 0])*1j)
                        U[2, 0] = (m + self.sf / 2)*exp(angle(U[2, 0])*1j)

                watermarked[i * self.R1:(i + 1) * self.R1,
                            j * self.R2:(j + 1) * self.R2] = (U * s) @ Vh

        return watermarked

    def extract_watermark(self, watermarked_image):
        R = self.IMAGE_TO_WATERMARK_RATIO
        watermark = empty(shape=(watermarked_image.shape[0] // R,
                                 watermarked_image.shape[1] // R))
        for i in range(0, watermark.shape[0]):
            for j in range(0, watermark.shape[1]):
                U, s, Vh = svd(
                    watermarked_image[i * self.R1:(i + 1) * self.R1,
                                      j * self.R2:(j + 1) * self.R2])
                # complexity = s[s != 0].size / s.size

                positive = abs(U[2, 0]) - abs(U[1, 0]) >= self.sf / 2

                # negative = abs(U[1, 0]) - abs(U[2, 0]) >= self.sf / 2
                # assert positive or negative

                watermark[i, j] = positive

        return watermark
