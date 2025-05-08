# Watermarking

[![DOI](https://zenodo.org/badge/444999271.svg)](https://zenodo.org/badge/latestdoi/444999271)  
[![Pytest with python-latest](https://github.com/naaci/watermarking/actions/workflows/test-with-python-latest.yml/badge.svg)](https://github.com/naaci/watermarking/actions/workflows/test-with-python-latest.yml)  
[![Pytest with python-3.11](https://github.com/naaci/watermarking/actions/workflows/test-with-python-3.11.yml/badge.svg)](https://github.com/naaci/watermarking/actions/workflows/test-with-python-3.11.yml)

---

## Overview
This repository contains implementations of various digital image watermarking schemes for grayscale images, as proposed by different authors.  
Each module defines a class named `watermarking` with the following methods:
- `add_watermark`: Adds a watermark to an image.
- `extract_watermark`: Extracts a watermark from an image.
- Some modules may also include `extract_watermarks` for handling multiple watermarks.

---

## Installation
To get started, install the required Python dependencies:
```bash
pip install -r requirements.txt

## Testing
`pytest`
