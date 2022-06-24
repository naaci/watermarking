"""
Sun, R., Sun, H., & Yao, T. (2002). 
A SVD and quantization based semi-fragile watermarking technique for image authentication. 
International Conference on Signal Processing Proceedings, ICSP, 2, 1592–1595. 
https://doi.org/10.1109/ICOSP.2002.1180102
"""

from numpy import empty, empty_like, mod
from numpy.linalg import svd


class Watermarker:
    IMAGE_TO_WATERMARK_RATIO = 4

    def __init__(self, scale_factor=2**(-4)) -> None:
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
                z = mod(s[0], self.sf)
                if watermark_image[i, j] < .5:
                    if z >= self.sf * 3 / 4:
                        s[0] += -z + self.sf * 5 / 4
                    else:
                        s[0] += -z + self.sf / 4
                else:
                    if z >= self.sf / 4:
                        s[0] += -z + self.sf * 3 / 4
                    else:
                        s[0] += -z - self.sf / 4

                watermarked[i * self.R1:(i + 1) * self.R1,
                            j * self.R2:(j + 1) * self.R2] = (U * s) @ Vh

        return watermarked

    def extract_watermark(self, watermarked_image):
        # R = self.IMAGE_TO_WATERMARK_RATIO
        watermark = empty(shape=(watermarked_image.shape[0] // self.R1,
                                 watermarked_image.shape[1] // self.R2))
        for i in range(0, watermark.shape[0]):
            for j in range(0, watermark.shape[1]):
                s = svd(watermarked_image[i * self.R1:(i + 1) * self.R1,
                                          j * self.R2:(j + 1) * self.R2],
                        compute_uv=False)
                z = mod(s[0], self.sf)
                watermark[i, j] = z > self.sf / 2
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
