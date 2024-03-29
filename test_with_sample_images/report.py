import io
from datetime import datetime
from pathlib import Path
from sys import path, stderr

from fpdf import FPDF
from numpy import asarray, ascontiguousarray, clip
from numpy.random import RandomState
from PIL import Image
from scipy.ndimage import gaussian_filter
from simplejpeg import decode_jpeg, encode_jpeg
from skimage import color, exposure, transform
from tifffile import tifffile

import arnold, metrics
from importlib.machinery import SourceFileLoader
import os

os.system("")

# import imagecodecs
# import imageio
# import napari
# from cv2 import GaussianBlur

cwd = Path(__file__).parent
path.append(str(cwd.parent / "watermarking"))

# seterr(all="raise")
k, p, q = 5, 12, 13
p = q = 1
# k = 0


def report(friendyname, name, bitdepth=8, format="png"):
    def decorator(attack):
        def wrapper(self, *args, **kwargs):
            watermarked__ = self._pre_process(
                watermarked_ := attack(
                    self,
                    watermarked := self._post_process(
                        self.watermarked,
                        bitdepth=bitdepth,
                    ),
                    *args,
                    **kwargs,
                ),
                bitdepth=bitdepth,
            )

            psnr = self.PSNR(self.host, watermarked__)
            ssim = self.SSIM(self.host, watermarked__)

            watermarked___ = self.truncate(watermarked_, bitdepth=bitdepth)

            self.pdf2.add_page(format=(watermarked___.shape[1] * 2 + 4, 12))
            self.pdf2.cell(0, txt=name.format(*args, **kwargs), align="C")

            self.pdf2.add_page(
                format=(
                    watermarked___.shape[1] * 2 + 4,
                    watermarked___.shape[0] + 12 * 2,
                ),
            )

            self.pdf2.start_section(
                f"{friendyname.format(*args, **kwargs)} PSNR:{psnr:3.1f} dB"
            )

            self.pdf2.image(
                Image.fromarray(watermarked___),
                x=0,
                # y=(self.pdf2.h-12-watermarked_.shape[0])/2,
                y=0,
                alt_text=friendyname.format(*args, **kwargs),
                # type=format,
            )
            self.pdf2.y = self.pdf2.h - 12
            self.pdf2.cell(self.pdf2.w // 2, txt=f"PSNR:{psnr:.1f}dB", align="C")

            if hasattr(self.watermarker, "extract_watermarks"):
                ncc_str = ""
                ssim_str = ""
                for i, watermark_extracted in enumerate(
                    self._extract_watermarks(watermarked__)
                ):
                    ncc = self.NCC(self.watermark, watermark_extracted)
                    ssim = self.SSIM(self.watermark, watermark_extracted)

                    self.pdf2.start_section(f"NCC:{ncc:6.1%} SSIM:{ssim:6.1%}", level=1)

                    watermark_extracted_ = self.truncate(
                        self._post_process(watermark_extracted)
                    )

                    a = self.pdf2.w // 4 - watermark_extracted_.shape[0]
                    b = (self.pdf2.h - 12 * 2) // 2 - watermark_extracted_.shape[1]

                    padx = (
                        a // 2 + (a + watermark_extracted_.shape[1] + 2) * (i % 2) + 2
                    )
                    pady = b // 2 + (b + watermark_extracted_.shape[0] + 12) * (i // 2)

                    self.pdf2.image(
                        Image.fromarray(watermark_extracted_),
                        x=self.pdf2.w // 2 + padx,
                        y=pady,
                    )
                    self.pdf2.y = pady + watermark_extracted_.shape[1]
                    self.pdf2.x = padx + watermarked.shape[0]
                    self.pdf2.cell(
                        watermark_extracted_.shape[1], txt=f"NCC:{ncc:.1%}", align="C"
                    )

                    ncc_str += ("&" if i % 2 else r"\\") + f" {100*ssim:.1f}%"
                    ssim_str += ("&" if i % 2 else r"\\") + f" {100*ncc:.1f}%"
            else:
                watermark_extracted_ = self.truncate(
                    self._post_process(
                        watermark_extracted := self._extract_watermark(watermarked__)
                    )
                )

                if not watermark_extracted.imag.any():
                    self.watermark = self.watermark.real

                ncc = self.NCC(self.watermark, watermark_extracted)
                ssim = self.SSIM(self.watermark, watermark_extracted)
                self.pdf2.start_section(f"NCC:{ncc:6.1%} SSIM:{ssim:6.1%}", level=1)

                padx = (self.pdf2.w // 2 - watermark_extracted.shape[1]) // 2 + 4
                pady = (self.pdf2.h - 12 * 2 - watermark_extracted.shape[0]) // 2

                self.pdf2.image(
                    Image.fromarray(watermark_extracted_),
                    x=self.pdf2.w // 2 + padx,
                    y=pady,
                )
                self.pdf2.cell(self.pdf2.w // 2, txt=f"NCC:{ncc:.1%}", align="C")

        return wrapper

    return decorator


class Report_Generator:
    SF = 1

    # sf = 2**-5

    def truncate(self, watermarked, bitdepth=8):
        return (clip(watermarked, 0, 1) * (2**bitdepth - 1)).astype(f"uint{bitdepth}")

    def _pre_process(self, image, bitdepth=None):
        if len(image.shape) == 3:
            return color.rgb2gray(image)
        return image

    def _post_process(self, image, bitdepth=None):
        return image

    def imread(self, path, size=None):
        image = asarray(Image.open(path))
        # image = imageio.imread(path)
        image = image / (2 ** (8 * image.dtype.itemsize) - 1)
        if size is not None:
            image = transform.resize(image, size)
        return self._pre_process(image)

    # def imwrite(self, path, image, *args, bitdepth=8, **kvargs):
    #     # https://github.com/numpy/numpy/issues/15630
    #     # np.clip with complex input is untested and has odd behavior #15630
    #     image2 = (clip(self._post_process(image, bitdepth=bitdepth), 0, 1) *
    #               (2**bitdepth - 1)).astype(f"uint{bitdepth}")
    #     # working_path /
    #     return imageio.imsave(path, image2, *args, **kvargs)

    def MSE(self, image1, image2):
        return metrics.MSE(
            self._post_process(image1),
            self._post_process(image2),
        )

    def PSNR(self, image1, image2):
        return metrics.PSNR(
            self._post_process(image1),
            self._post_process(image2),
        )

    def NCC(self, image1, image2):
        return metrics.NCC(
            self._post_process(image1),
            self._post_process(image2),
        )

    def SSIM(self, image1, image2):
        return metrics.SSIM(
            self._post_process(image1),
            self._post_process(image2),
        )

    ###################################################

    @report("PNG", "png")
    def _test_in_memory(self, watermarked):
        return watermarked

    @report("PNG ({bitdepth} bit)", "png{bitdepth}")
    def _test_bitdepth(self, watermarked, bitdepth=8):
        io_buf = io.BytesIO()
        tifffile.imwrite(io_buf, self.truncate(watermarked, bitdepth=bitdepth))
        io_buf.seek(0)
        attacked = asarray(tifffile.imread(io_buf))
        return attacked / (2 ** (8 * attacked.dtype.itemsize) - 1)

    @report("{compression} Sıkıştırma", "{compression}")
    def _test_lossy_compression(self, watermarked, compression, *args):
        io_buf = io.BytesIO()
        # tifffile.imwrite(io_buf,
        #                  self.truncate(watermarked),
        #                  compression=compression)
        # io_buf.seek(0)
        # attacked = asarray(tifffile.imread(io_buf))
        Image.fromarray(watermarked).save(
            io_buf,
            format=compression,
        )
        attacked = asarray(Image.open(io_buf))
        return attacked / (2 ** (8 * attacked.dtype.itemsize) - 1)

    @report("JPEG (Q={q})", "jpg{q}", format="jpeg")
    def _test_jpeg_compression(self, watermarked, q=95):
        # io_buf = io.BytesIO()
        if len(watermarked.shape) == 2:
            watermarked = watermarked.reshape(*watermarked.shape, 1)
        colorspace = "GRAY" if watermarked.shape[2] == 1 else "RGB"
        watermarked_ = decode_jpeg(
            j := encode_jpeg(
                a := ascontiguousarray(self.truncate(watermarked)),
                colorspace=colorspace,
                quality=q,
            ),
        )
        # tifffile.imwrite(io_buf,
        #                  self.truncate(watermarked),
        #                  compression="jpeg")
        # io_buf.seek(0)
        # attacked = asarray(tifffile.imread(io_buf))
        # self.watermarked_pil_image = Image.fromarray(
        #     a := self.truncate(self.watermarked, bitdepth=8))

        # self.watermarked_pil_image.save(
        #     io_buf,
        #     format="jpeg",
        #     quality=q,
        # )
        # attacked = asarray(Image.open(io_buf))
        return watermarked_ / (2 ** (8 * watermarked_.dtype.itemsize) - 1)

    @report("FPP0 Testi", "ffp0")
    def _test_false_pasitive_0(self, watermarked):
        return self._post_process(self.host)

    @report("FPP1 Testi", "ffp1")
    def _test_false_pasitive_1(self, watermarked):
        watermarked = watermarked.copy()
        self._add_watermark(watermark=self.watermark2)
        return watermarked

    @report("Histogram Eşitleme", "hist")
    def histogram_equalize(self, watermarked):
        return exposure.equalize_hist(watermarked)

    @report("{angle} Döndürme", "rotate{angle}")
    def _test_rotation(self, watermarked, angle=1):
        # return asarray(self.watermarked_pil_image.rotate(angle))
        return transform.rotate(watermarked, angle)

    @report("Kırpma ({r})", "crop{r}")
    def _test_crop(self, watermarked, r=16):
        watermarked_ = watermarked.copy()
        watermarked_[: watermarked_.shape[0] // r] = 0
        watermarked_[-watermarked_.shape[0] // r :] = 0
        watermarked_[:, : watermarked_.shape[1] // r] = 0
        watermarked_[:, -watermarked_.shape[1] // r :] = 0
        return watermarked_

    @report("Gaussian Blur ({sigma})", "blur{sigma}")
    def _test_gaussian_blur(self, watermarked, sigma=1):
        # return asarray(
        #     self.watermarked_pil_image.filter(
        #         ImageFilter.GaussianBlur(radius=radius)))
        return gaussian_filter(watermarked, sigma=sigma)
        # return GaussianBlur(watermarked, ksize=ksize)

    @report("Gaussian Noise ({weight:.0%})", "noise{weight:.0f}")
    def _test_gaussian_noise(self, watermarked, weight=0.3):
        random = RandomState(11052022)
        noise = random.random(size=watermarked.shape)
        return watermarked + noise * weight

    ###################################################

    def _add_watermark(self, watermark=None):
        if watermark is None:
            watermark = self.watermark

        k, p, q = self.at
        if p == q == 1:
            watermark, self.KEY = arnold.arnold_transform1(watermark, k)
        else:
            watermark, self.KEY = arnold.arnold_transform(watermark, k, p, q)

        self.watermarked = self.watermarker.add_watermark(self.host, watermark)

    def _extract_watermark(self, watermarked):
        watermark = self.watermarker.extract_watermark(watermarked)
        return arnold.coordinate_transform(watermark, self.KEY)

    def _extract_watermarks(self, watermarked):
        return (
            arnold.coordinate_transform(image, self.KEY)
            for image in self.watermarker.extract_watermarks(watermarked)
        )

    ###################################################

    def __init__(
        self,
        watermarker,
        host_path,
        watermark_path,
        watermark2_path,
        working_path,
        k=0,
        p=1,
        q=1,
    ):
        self.at = k, p, q

        working_path.parent.mkdir(parents=True, exist_ok=True)
        self.working_path = working_path

        self.watermarker = watermarker

        self.host = self.imread(host_path)

        self.watermark = self.imread(
            watermark_path,
            size=(
                self.host.shape[0] // watermarker.IMAGE_TO_WATERMARK_RATIO,
                self.host.shape[1] // watermarker.IMAGE_TO_WATERMARK_RATIO,
            ),
        )

        self.watermark2 = self.imread(
            watermark2_path,
            size=(
                self.host.shape[0] // watermarker.IMAGE_TO_WATERMARK_RATIO,
                self.host.shape[1] // watermarker.IMAGE_TO_WATERMARK_RATIO,
            ),
        )

    def tests(self, module):
        self._add_watermark()

        self.pdf2 = FPDF(
            unit="pt",
            #  format=(
            #      self.host.shape[1] * 2 + 4,
            #      self.host.shape[0] + 12 * 2,
            #  ),
        )
        self.pdf2.set_display_mode(
            zoom="real",
            layout="single",
        )
        self.pdf2.set_creation_date(datetime.fromisoformat("2022-06-27"))
        self.pdf2.set_compression(compress=True)
        self.pdf2.set_author("Naci ER")
        self.pdf2.set_subject("PhD thesis attachment")
        self.pdf2.set_title(module)
        self.pdf2.set_font("times")
        self.pdf2.set_font_size(12)
        self.pdf2.set_margin(0)
        self.pdf2.set_auto_page_break(False)

        self._test_in_memory()

        self._test_bitdepth(bitdepth=16)
        self._test_bitdepth(bitdepth=8)

        self._test_jpeg_compression(q=90)
        self._test_jpeg_compression(q=60)
        self._test_jpeg_compression(q=30)

        # self._test_lossy_compression(compression="jp2")
        # self._test_lossy_compression(compression="webp")

        self.histogram_equalize()

        self._test_rotation(angle=1)
        self._test_rotation(angle=30)
        self._test_rotation(angle=90)

        self._test_crop(r=16)

        self._test_gaussian_blur(sigma=2)
        self._test_gaussian_noise(weight=0.1)

        self._test_false_pasitive_0()
        self._test_false_pasitive_1()

        self.pdf2.output(self.working_path.as_posix())


def _test_watermark(watermarker):
    host = "images/host_image.png"
    watermark1 = "images/watermark.png"
    watermark2 = "images/watermark2.png"
    # k = 0
    # p = 1
    # q = 1
    out = cwd / "results"
    out.mkdir(parents=True, exist_ok=True)

    if hasattr(Report_Generator, "sf"):
        watermarker.sf = Report_Generator.sf
    else:
        watermarker.sf *= Report_Generator.SF

    t = Report_Generator(
        watermarker,
        cwd / host,
        cwd / watermark1,
        cwd / watermark2,
        out
        / f"{watermarker.__module__}/{watermarker.__module__}.w={watermarker.sf:g}{f'.k={k}' if k else ''}{'.p={p}.q={q}' if k and p!=1 else ''}.pdf",
        k=k,
        p=p,
        q=q,
    )

    if not t.working_path.is_file():
        try:
            t.tests(watermarker.__module__)
        except Exception as err:
            stderr.write(f"\033[1;31m{str(err)}\033[0m")


def test_watermarks():
    for method in (cwd.parent / "watermarking").glob("*.py"):
        if method.name == "__init__.py":
            continue

        print("\033[1;32m", method.name, "\033[0m")

        _test_watermark(
            SourceFileLoader(method.name.removesuffix(".py"), method.as_posix())
            .load_module()
            .Watermarker(0.01),
        )


if __name__ == "__main__":
    test_watermarks()
