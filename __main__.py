from argparse import ArgumentParser

import numpy
import tifffile

# from Chandra2002 import IMAGE_TO_WATERMARK_RATIO, add_watermark
# from Jain2008 import IMAGE_TO_WATERMARK_RATIO, add_watermark
# from Lai2010 import IMAGE_TO_WATERMARK_RATIO, add_watermark
# from Liu2002 import IMAGE_TO_WATERMARK_RATIO, add_watermark
# from Liu2018 import IMAGE_TO_WATERMARK_RATIO, add_watermark
# from Liu2019 import IMAGE_TO_WATERMARK_RATIO, add_watermark
# from Sun2002 import IMAGE_TO_WATERMARK_RATIO, add_watermark
# from Yavuz2007 import IMAGE_TO_WATERMARK_RATIO, add_watermark
from Ganic2004 import Watermarker


def main():
    watermarker = Watermarker()

    host_image = tifffile.imread(args.host)
    host_image = host_image / numpy.iinfo(host_image.dtype).max

    watermark_image = tifffile.imread(args.watermark)
    watermark_image = watermark_image / numpy.iinfo(watermark_image.dtype).max

    watermark_image = watermark_image.reshape(
        host_image.shape[0] // watermarker.IMAGE_TO_WATERMARK_RATIO,
        watermark_image.shape[0] // host_image.shape[0] *
        watermarker.IMAGE_TO_WATERMARK_RATIO,
        host_image.shape[1] // watermarker.IMAGE_TO_WATERMARK_RATIO,
        watermark_image.shape[1] // host_image.shape[1] *
        watermarker.IMAGE_TO_WATERMARK_RATIO,
    ).max(3).max(1)

    watermarked = watermarker.add_watermark(
        host_image,
        watermark_image,
    )

    dtype = numpy.dtype(f"uint{args.bitdepth}")

    if args.extract:
        watermarked = tifffile.imread(args.watermarked)
        watermarked = watermarked / numpy.iinfo(watermarked.dtype).max
        tifffile.imwrite(args.extract,
                         (watermarker.extract_watermark(watermarked) *
                          numpy.iinfo(dtype).max).clip(
                              numpy.iinfo(dtype).min,
                              numpy.iinfo(dtype).max).astype(dtype),
                         compress=args.compress)

    else:
        tifffile.imwrite(args.watermarked,
                         (watermarked * numpy.iinfo(dtype).max).clip(
                             numpy.iinfo(dtype).min,
                             numpy.iinfo(dtype).max).astype(dtype),
                         compress=args.compress)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        'host',
        help='Original image to add watermark',
    )
    parser.add_argument(
        'watermark',
        help='Watermark image',
    )
    parser.add_argument(
        'watermarked',
        help='Watermarked image to extrct watermark',
    )
    parser.add_argument(
        '--extract',
        "-e",
        help='Extracted watermark',
    )
    parser.add_argument(
        '--bitdepth',
        "-b",
        default=8,
        type=int,
        help='bits used to store tiff image pixel',
    )
    parser.add_argument(
        '--compress',
        "-c",
        default=7,
        type=int,
    )

    args = parser.parse_args()
    main()
