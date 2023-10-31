from numpy import log10, sqrt
from skimage.metrics import structural_similarity


def MSE(image1, image2):
    return ((image1 - image2) ** 2).mean()


def PSNR(image1, image2):
    return 10 * log10(1 / MSE(image1, image2))


def NCC(image1, image2):
    return (image1 * image2).sum() / (
        sqrt((image1**2).sum()) * sqrt((image2**2).sum())
    )


def SSIM(image1, image2):
    return structural_similarity(
        image1,
        image2,
        channel_axis=-1,
        data_range=1.0,
    )
