"""
Liu, R., & Tan, T. (2002). 
An SVD-based watermarking scheme for protecting rightful ownership. 
IEEE Transactions on Multimedia, 4(1), 121–128. 
https://doi.org/10.1109/6046.985560
"""

# from numpy.linalg import svd
from scipy.linalg import diagsvd, svd


class Watermarker:
    IMAGE_TO_WATERMARK_RATIO = 1

    def __init__(self, scale_factor=.1) -> None:
        """Initiates watermarker object

        Args:
            scale_factor (float or array of floats, optional):
              Simple Scale Factor if the scale_factor is a float .
              Multiple Scale Factor if the scale_factor is an array of length k, 
                where k is the number of singular values.
              Defaults to 0.1
        """
        self.sf = scale_factor

    def add_watermark(self, host, watermark):
        U, s, Vh = svd(host, full_matrices=False)
        self.S = diagsvd(s, *host.shape[:2])
        D = self.S + watermark * self.sf
        self.U_w, s_w, self.Vh_w = svd(D, full_matrices=False)
        return (U * s_w) @ Vh

    def extract_watermark(self, watermarked_image):
        s_w = svd(watermarked_image, compute_uv=False)
        k = min(watermarked_image.shape[:2])
        D = (self.U_w * s_w) @ self.Vh_w
        return (D - self.S) / self.sf
