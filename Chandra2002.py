"""
Chandra, D. V. S. (2010). 
Digital image watermarking using singular value decomposition. 
The 2002 45th Midwest Symposium on Circuits and Systems, 2002. MWSCAS-2002., 3(3), III-264-III–267. 
https://doi.org/10.1109/MWSCAS.2002.1187023
"""

# from numpy.linalg import svd
from scipy.linalg import svd, diagsvd


class Watermarker:
    IMAGE_TO_WATERMARK_RATIO = 1

    def __init__(self, scale_factor=2**(-4)) -> None:
        """Initiates watermarker object

        Args:
            scale_factor (float or array of floats, optional):
              Simple Scale Factor if the scale_factor is a float .
              Multiple Scale Factor if the scale_factor is an array of length k, 
                where k is the number of singular values.
              Defaults to 2**(-4)
        """
        self.sf = scale_factor

    def add_watermark(self, host, watermark):
        U, self.s, Vh = svd(host, full_matrices=True)
        self.U_w, s_w, self.Vh_w = svd(watermark, full_matrices=True)
        s = self.s + self.sf * s_w
        return U @ diagsvd(s, *watermark.shape) @ Vh

    def extract_watermark(self, watermarked):
        s = svd(watermarked, compute_uv=False)
        s_w = (s - self.s) / self.sf
        return self.U_w @ diagsvd(s_w, *watermarked.shape) @ self.Vh_w
