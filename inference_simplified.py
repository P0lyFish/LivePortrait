# coding: utf-8

"""
The entrance of humans
"""

import cv2
import os
import os.path as osp
import subprocess
from liveportrait.config.argument_config import ArgumentConfig
from liveportrait.config.inference_config import InferenceConfig
from liveportrait.config.crop_config import CropConfig
from liveportrait.simplified_liveportrait_pipeline import SimplifiedLivePortraitPipeline
from liveportrait.utils.io import load_image_rgb
import numpy as np


def partial_fields(target_class, kwargs):
    return target_class(**{k: v for k, v in kwargs.items() if hasattr(target_class, k)})


def fast_check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except:
        return False


def fast_check_args(args: ArgumentConfig):
    if not osp.exists(args.source):
        raise FileNotFoundError(f"source info not found: {args.source}")
    if not osp.exists(args.driving):
        raise FileNotFoundError(f"driving info not found: {args.driving}")


def main():
    # set tyro theme
    args = ArgumentConfig()

    # specify configs for inference
    inference_cfg = partial_fields(InferenceConfig, args.__dict__)
    crop_cfg = partial_fields(CropConfig, args.__dict__)

    live_portrait_pipeline = SimplifiedLivePortraitPipeline(
        inference_cfg=inference_cfg,
        crop_cfg=crop_cfg
    )

    # Load data
    source = load_image_rgb('./assets/examples/source/s0.jpg')
    driver = load_image_rgb('./assets/examples/driving/d8.jpg')

    # run
    output = live_portrait_pipeline(source, driver)
    vis = np.hstack((source, driver, output))
    cv2.imwrite('debug.png', vis)


if __name__ == "__main__":
    main()
