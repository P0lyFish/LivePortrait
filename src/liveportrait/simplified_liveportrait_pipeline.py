# coding: utf-8

"""
Pipeline of LivePortrait (Human)
"""

import torch
torch.backends.cudnn.benchmark = True # disable CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR warning

import cv2; cv2.setNumThreads(0); cv2.ocl.setUseOpenCL(False)
import numpy as np
import os
import os.path as osp
from rich.progress import track

from .config.argument_config import ArgumentConfig
from .config.inference_config import InferenceConfig
from .config.crop_config import CropConfig
from .utils.cropper import Cropper
from .utils.camera import get_rotation_matrix
from .utils.video import images2video, concat_frames, get_fps, add_audio_to_video, has_audio_stream
from .utils.crop import prepare_paste_back, paste_back
from .utils.io import load_image_rgb, load_video, resize_to_limit, dump, load
from .utils.helper import mkdir, basename, dct2device, is_video, is_template, remove_suffix, is_image, is_square_video, calc_motion_multiplier
from .utils.filter import smooth
from .utils.rprint import rlog as log
# from .utils.viz import viz_lmk
from .live_portrait_wrapper import LivePortraitWrapper


def make_abs_path(fn):
    return osp.join(osp.dirname(osp.realpath(__file__)), fn)


class SimplifiedLivePortraitPipeline(object):

    def __init__(self, inference_cfg: InferenceConfig, crop_cfg: CropConfig):
        self.live_portrait_wrapper: LivePortraitWrapper = LivePortraitWrapper(inference_cfg=inference_cfg)
        self.cropper: Cropper = Cropper(crop_cfg=crop_cfg)

    def make_motion_template(self, I_lst, c_eyes_lst, c_lip_lst, **kwargs):
        n_frames = I_lst.shape[0]
        template_dct = {
            'n_frames': n_frames,
            'output_fps': kwargs.get('output_fps', 25),
            'motion': [],
            'c_eyes_lst': [],
            'c_lip_lst': [],
        }

        for i in track(range(n_frames), description='Making motion templates...', total=n_frames):
            # collect s, R, δ and t for inference
            I_i = I_lst[i]
            x_i_info = self.live_portrait_wrapper.get_kp_info(I_i)
            x_s = self.live_portrait_wrapper.transform_keypoint(x_i_info)
            R_i = get_rotation_matrix(x_i_info['pitch'], x_i_info['yaw'], x_i_info['roll'])

            item_dct = {
                'scale': x_i_info['scale'].cpu().numpy().astype(np.float32),
                'R': R_i.cpu().numpy().astype(np.float32),
                'exp': x_i_info['exp'].cpu().numpy().astype(np.float32),
                't': x_i_info['t'].cpu().numpy().astype(np.float32),
                'kp': x_i_info['kp'].cpu().numpy().astype(np.float32),
                'x_s': x_s.cpu().numpy().astype(np.float32),
            }

            template_dct['motion'].append(item_dct)

            c_eyes = c_eyes_lst[i].astype(np.float32)
            template_dct['c_eyes_lst'].append(c_eyes)

            c_lip = c_lip_lst[i].astype(np.float32)
            template_dct['c_lip_lst'].append(c_lip)

        return template_dct

    def __call__(self, source, driver):
        # for convenience
        inf_cfg = self.live_portrait_wrapper.inference_cfg
        device = self.live_portrait_wrapper.device
        crop_cfg = self.cropper.crop_cfg

        ######## process driving info ########
        flag_load_from_template = is_template(args.driving)
        driving_rgb_crop_256x256_lst = None
        wfp_template = None

        ######## make motion template ########
        driving_lmk_crop = self.cropper.calc_lmks_from_cropped_video([driver])[0]
        driver = cv2.resize(driver, (256, 256))
        #######################################

        c_d_eyes_lst, c_d_lip_lst = self.live_portrait_wrapper.calc_ratio([driving_lmk_crop])
        # # save the motion template
        I_d_lst = self.live_portrait_wrapper.prepare_videos([driver])
        driving_template_dct = self.make_motion_template(I_d_lst, c_d_eyes_lst, c_d_lip_lst, output_fps=24)

        # wfp_template = remove_suffix(args.driving) + '.pkl'
        # dump(wfp_template, driving_template_dct)
        # log(f"Dump motion template to {wfp_template}")
        # if not flag_is_driving_video:
        #     c_d_eyes_lst = c_d_eyes_lst*n_frames
        #     c_d_lip_lst = c_d_lip_lst*n_frames

        ######## prepare for pasteback ########
        R_d_0, x_d_0_info = None, None
        flag_normalize_lip = inf_cfg.flag_normalize_lip  # not overwrite
        flag_source_video_eye_retargeting = inf_cfg.flag_source_video_eye_retargeting  # not overwrite
        lip_delta_before_animation, eye_delta_before_animation = None, None

        ######## process source info ########
        crop_info = self.cropper.crop_source_image(source, crop_cfg)
        if crop_info is None:
            return None
        source_lmk = crop_info['lmk_crop']
        img_crop_256x256 = crop_info['img_crop_256x256']

        I_s = self.live_portrait_wrapper.prepare_source(img_crop_256x256)
        x_s_info = self.live_portrait_wrapper.get_kp_info(I_s)
        x_c_s = x_s_info['kp']
        R_s = get_rotation_matrix(x_s_info['pitch'], x_s_info['yaw'], x_s_info['roll'])
        f_s = self.live_portrait_wrapper.extract_feature_3d(I_s)
        x_s = self.live_portrait_wrapper.transform_keypoint(x_s_info)

        # let lip-open scalar to be 0 at first
        if flag_normalize_lip and inf_cfg.flag_relative_motion and source_lmk is not None:
            c_d_lip_before_animation = [0.]
            combined_lip_ratio_tensor_before_animation = self.live_portrait_wrapper.calc_combined_lip_ratio(c_d_lip_before_animation, source_lmk)
            if combined_lip_ratio_tensor_before_animation[0][0] >= inf_cfg.lip_normalize_threshold:
                lip_delta_before_animation = self.live_portrait_wrapper.retarget_lip(x_s, combined_lip_ratio_tensor_before_animation)

        if inf_cfg.flag_pasteback and inf_cfg.flag_do_crop and inf_cfg.flag_stitching:
            mask_ori_float = prepare_paste_back(inf_cfg.mask_crop, crop_info['M_c2o'], dsize=(source.shape[1], source.shape[0]))

        x_d_i_info = driving_template_dct['motion'][0]
        x_d_i_info = dct2device(x_d_i_info, device)
        R_d_i = x_d_i_info['R'] if 'R' in x_d_i_info.keys() else x_d_i_info['R_d']  # compatible with previous keys

        R_d_0 = R_d_i
        x_d_0_info = x_d_i_info.copy()

        delta_new = x_s_info['exp'].clone()
        if inf_cfg.flag_relative_motion:
            R_new = (R_d_i @ R_d_0.permute(0, 2, 1)) @ R_s
            delta_new = x_s_info['exp'] + (x_d_i_info['exp'] - torch.from_numpy(inf_cfg.lip_array).to(dtype=torch.float32, device=device))
            scale_new = x_s_info['scale'] * (x_d_i_info['scale'] / x_d_0_info['scale'])
            t_new = x_s_info['t'] + (x_d_i_info['t'] - x_d_0_info['t'])
        else:
            R_new = R_d_i
            for idx in [1,2,6,11,12,13,14,15,16,17,18,19,20]:
                delta_new[:, idx, :] = x_d_i_info['exp'][:, idx, :]
            delta_new[:, 3:5, 1] = x_d_i_info['exp'][:, 3:5, 1]
            delta_new[:, 5, 2] = x_d_i_info['exp'][:, 5, 2]
            delta_new[:, 8, 2] = x_d_i_info['exp'][:, 8, 2]
            delta_new[:, 9, 1:] = x_d_i_info['exp'][:, 9, 1:]
            scale_new = x_s_info['scale']
            t_new = x_d_i_info['t']

        t_new[..., 2].fill_(0)  # zero tz
        x_d_i_new = scale_new * (x_c_s @ R_new + delta_new) + t_new

        # Algorithm 1:
        if not inf_cfg.flag_stitching and not inf_cfg.flag_eye_retargeting and not inf_cfg.flag_lip_retargeting:
            # without stitching or retargeting
            if flag_normalize_lip and lip_delta_before_animation is not None:
                x_d_i_new += lip_delta_before_animation
            if flag_source_video_eye_retargeting and eye_delta_before_animation is not None:
                x_d_i_new += eye_delta_before_animation
            else:
                pass
        elif inf_cfg.flag_stitching and not inf_cfg.flag_eye_retargeting and not inf_cfg.flag_lip_retargeting:
            # with stitching and without retargeting
            if flag_normalize_lip and lip_delta_before_animation is not None:
                x_d_i_new = self.live_portrait_wrapper.stitching(x_s, x_d_i_new) + lip_delta_before_animation
            else:
                x_d_i_new = self.live_portrait_wrapper.stitching(x_s, x_d_i_new)
            if flag_source_video_eye_retargeting and eye_delta_before_animation is not None:
                x_d_i_new += eye_delta_before_animation
        else:
            eyes_delta, lip_delta = None, None
            if inf_cfg.flag_eye_retargeting and source_lmk is not None:
                c_d_eyes_i = c_d_eyes_lst[0]
                combined_eye_ratio_tensor = self.live_portrait_wrapper.calc_combined_eye_ratio(c_d_eyes_i, source_lmk)
                # ∆_eyes,i = R_eyes(x_s; c_s,eyes, c_d,eyes,i)
                eyes_delta = self.live_portrait_wrapper.retarget_eye(x_s, combined_eye_ratio_tensor)
            if inf_cfg.flag_lip_retargeting and source_lmk is not None:
                c_d_lip_i = c_d_lip_lst[0]
                combined_lip_ratio_tensor = self.live_portrait_wrapper.calc_combined_lip_ratio(c_d_lip_i, source_lmk)
                # ∆_lip,i = R_lip(x_s; c_s,lip, c_d,lip,i)
                lip_delta = self.live_portrait_wrapper.retarget_lip(x_s, combined_lip_ratio_tensor)

            if inf_cfg.flag_relative_motion:  # use x_s
                x_d_i_new = x_s + \
                    (eyes_delta if eyes_delta is not None else 0) + \
                    (lip_delta if lip_delta is not None else 0)
            else:  # use x_d,i
                x_d_i_new = x_d_i_new + \
                    (eyes_delta if eyes_delta is not None else 0) + \
                    (lip_delta if lip_delta is not None else 0)

            if inf_cfg.flag_stitching:
                x_d_i_new = self.live_portrait_wrapper.stitching(x_s, x_d_i_new)

        x_d_i_new = x_s + (x_d_i_new - x_s) * inf_cfg.driving_multiplier
        out = self.live_portrait_wrapper.warp_decode(f_s, x_s, x_d_i_new)
        I_p_i = self.live_portrait_wrapper.parse_output(out['out'])[0]

        if inf_cfg.flag_pasteback and inf_cfg.flag_do_crop and inf_cfg.flag_stitching:
            # TODO: the paste back procedure is slow, considering optimize it using multi-threading or GPU
            I_p_pstbk = paste_back(I_p_i, crop_info['M_c2o'], source, mask_ori_float)

        ######### build the final concatenation result #########
        # driving frame | source frame | generation
        if inf_cfg.flag_paste:
            return I_p_pstbk
        else:
            return I_p_i
