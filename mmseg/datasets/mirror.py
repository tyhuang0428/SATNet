import os.path as osp

import mmcv
from mmcv.utils import print_log
from PIL import Image
import numpy as np

from mmseg.core import get_eval
from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class MirrorDataset(CustomDataset):
    """Cityscapes dataset.

    The ``img_suffix`` is fixed to '_leftImg8bit.png' and ``seg_map_suffix`` is
    fixed to '_gtFine_labelTrainIds.png' for Cityscapes dataset.
    """

    CLASSES = ('background', 'mirror')

    PALETTE = np.array([[0, 0, 0], [255, 255, 255]], dtype=np.uint8)

    def __init__(self, **kwargs):
        super(MirrorDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            **kwargs)
        self.label_map = {255: 1}

    def results2img(self, results, imgfile_prefix):
        mmcv.mkdir_or_exist(imgfile_prefix)
        result_files = []
        prog_bar = mmcv.ProgressBar(len(self))
        for idx in range(len(self)):
            result = results[idx]
            filename = self.img_infos[idx]['filename']
            basename = osp.splitext(osp.basename(filename))[0]
            png_filename = osp.join(imgfile_prefix, f'{basename}.png')

            output = Image.fromarray(result.astype(np.uint8)).convert('P')
            output.putpalette(self.PALETTE)
            output.save(png_filename)
            result_files.append(png_filename)
            prog_bar.update()

        return result_files

    def format_results(self, results, imgfile_prefix=None, to_label_id=False):
        if imgfile_prefix is None:
            imgfile_prefix = 'results/'
        result_files = self.results2img(results, imgfile_prefix)
        return result_files

    def evaluate(self, results, metric='mIoU', logger=None, **kwargs):
        eval_results = {}
        gt_seg_maps = self.get_gt_seg_maps()
        mean_ACC, mean_IOU, F, mean_MAE, mean_BER = get_eval(results, gt_seg_maps)
        summary_str = ("{:20} {:.2f} \n{:20} {:.3f} \n{:20} {:.3f} \n{:20} {:.3f} \n{:20} {:.2f}\n".
                       format("mean_IOU", mean_IOU, "mean_ACC", mean_ACC, "F", F,
                              "mean_MAE", mean_MAE, "mean_BER", mean_BER))
        print_log(summary_str, logger)

        eval_results['mean_ACC'] = mean_ACC
        eval_results['mean_IOU'] = mean_IOU
        eval_results['F'] = F
        eval_results['mean_MAE'] = mean_MAE
        eval_results['maen_BER'] = mean_BER

        return eval_results
