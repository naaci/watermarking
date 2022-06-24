"""
Xing, Y., & Tan, J. (2007). 
A Color Watermarking Scheme Based on Block-SVD and Arnold Transformation. 
Second Workshop on Digital Media and Its Application in Museum & Heritages (DMAMH 2007), 3–8. 
https://doi.org/10.1109/DMAMH.2007.15
"""

from numpy import empty, empty_like
from numpy.linalg import svd


class Watermarker:
    IMAGE_TO_WATERMARK_RATIO = 4

    def __init__(self, scale_factor=2**(-4)) -> None:
        self.sf = scale_factor

    def add_watermark(self, host_image, watermark_image):
        self.R1 = host_image.shape[0] // watermark_image.shape[0]
        self.R2 = host_image.shape[1] // watermark_image.shape[1]

        watermarked = empty_like(host_image)
        self.s = empty_like(watermark_image)
        for i in range(0, watermark_image.shape[0]):
            for j in range(0, watermark_image.shape[1]):
                U, s, Vh = svd(host_image[i * self.R1:(i + 1) * self.R1,
                                          j * self.R2:(j + 1) * self.R2],
                               full_matrices=False)
                self.s[i, j] = s[0]
                s[0] += self.sf * watermark_image[i, j]
                watermarked[i * self.R1:(i + 1) * self.R1,
                            j * self.R2:(j + 1) * self.R2] = (U * s) @ Vh

        return watermarked

    def extract_watermark(self, watermarked_image):
        R = self.IMAGE_TO_WATERMARK_RATIO
        watermark = empty(shape=(watermarked_image.shape[0] // R,
                                 watermarked_image.shape[1] // R))
        for i in range(0, watermark.shape[0]):
            for j in range(0, watermark.shape[1]):
                s = svd(watermarked_image[i * self.R1:(i + 1) * self.R1,
                                          j * self.R2:(j + 1) * self.R2],
                        compute_uv=False)
                watermark[i, j] = (s[0] - self.s[i, j]) / self.sf
        return watermark


if __name__ == "__main__":
    from numpy import asarray, isclose, random
    watermarker = Watermarker(.01)
    host = random.random((500, 600))
    watermark = random.random(
        asarray(host.shape) // watermarker.IMAGE_TO_WATERMARK_RATIO)
    watermark[watermark >= .5] = 1
    watermark[watermark < .5] = 0
    watermarked = watermarker.add_watermark(host, watermark)
    watermark_ = watermarker.extract_watermark(watermarked)
    c = isclose(watermark, watermark_)
    print("max err:", abs(watermark - watermark_).max())
    print(c.all() or f"{watermark[c].size / watermark.size:>.2%} True")
