[![DOI](https://zenodo.org/badge/444999271.svg)](https://zenodo.org/badge/latestdoi/444999271)
[![Pytest with python-latest](https://github.com/naaci/watermarking/actions/workflows/test-with-python-latest.yml/badge.svg)](https://github.com/naaci/watermarking/actions/workflows/test-with-python-latest.yml)
[![Pytest with python-3.11](https://github.com/naaci/watermarking/actions/workflows/test-with-python-3.11.yml/badge.svg)](https://github.com/naaci/watermarking/actions/workflows/test-with-python-3.11.yml)

# installation
You can install python dependencies with
`pip install -r requirements.txt`

# watermarking
Implementations of some digital image watermarking schemes for grayscale images proposed by various authors. 

Each module defines a class named `watermarking` with two methods namely `add_watermark` and `extract_watermark`. Some modules `watermarking` class has additional method `extract_watermarks`.
