from det3d.core.bbox import box_torch_ops
from ..registry import DETECTORS
from .base import BaseDetector
from .. import builder
import torch 
from torch import nn 

@DETECTORS.register_module
class TwoStageDetector(BaseDetector):
    def __init__(
        self,
        first_stage_cfg,
        roi_head, 
        NMS_POST_MAXSIZE,
        freeze=False,
        **kwargs
    ):
        super(TwoStageDetector, self).__init__()
        self.single_det = builder.build_detector(first_stage_cfg, **kwargs)
        self.NMS_POST_MAXSIZE = NMS_POST_MAXSIZE

        if freeze:
            print("Freeze First Stage Network")
            # we train the model in two steps 
            self.single_det = self.single_det.freeze()
        self.bbox_head = self.single_det.bbox_head

        self.roi_head = builder.build_roi_head(roi_head)

    def combine_loss(self, one_stage_loss, roi_loss, tb_dict):
        one_stage_loss['loss'][0] += (roi_loss)

        for i in range(len(one_stage_loss['loss'])):
            one_stage_loss['roi_reg_loss'].append(tb_dict['rcnn_loss_reg'])
            one_stage_loss['roi_cls_loss'].append(tb_dict['rcnn_loss_cls'])

        return one_stage_loss

    def reorder_first_stage_pred_and_feature(self, first_pred, example):
        batch_size = len(first_pred)
        box_length = first_pred[0]['box3d_lidar'].shape[1]

        rois = first_pred[0]['box3d_lidar'].new_zeros((batch_size, 
            self.NMS_POST_MAXSIZE, box_length 
        ))
        roi_scores = first_pred[0]['scores'].new_zeros((batch_size,
            self.NMS_POST_MAXSIZE
        ))
        roi_labels = first_pred[0]['label_preds'].new_zeros((batch_size,
            self.NMS_POST_MAXSIZE), dtype=torch.long
        )

        for i in range(batch_size):
            # basically move rotation to position 6, so now the box is 7 + C . C is 2 for nuscenes to
            # include velocity target

            box_preds = first_pred[i]['box3d_lidar']
            num_obj = box_preds.shape[0]

            if self.roi_head.code_size == 9:
                # x, y, z, w, l, h, rotation_y, velocity_x, velocity_y
                box_preds = box_preds[:, [0, 1, 2, 3, 4, 5, 8, 6, 7]]

            rois[i, :num_obj] = box_preds
            roi_labels[i, :num_obj] = first_pred[i]['label_preds'] + 1
            roi_scores[i, :num_obj] = first_pred[i]['scores']

        example['rois'] = rois 
        example['roi_labels'] = roi_labels 
        example['roi_scores'] = roi_scores
        example['has_class_labels']= True 

        return example 

    def post_process(self, batch_dict):
        batch_size = batch_dict['batch_size']
        pred_dicts = [] 

        for index in range(batch_size):
            box_preds = batch_dict['batch_box_preds'][index]
            cls_preds = batch_dict['batch_cls_preds'][index]  # this is the predicted iou 
            label_preds = batch_dict['roi_labels'][index]

            if box_preds.shape[-1] == 9:
                # move rotation to the end (the create submission file will take elements from 0:6 and -1) 
                box_preds = box_preds[:, [0, 1, 2, 3, 4, 5, 7, 8, 6]]

            scores = torch.sqrt(torch.sigmoid(cls_preds).reshape(-1) * batch_dict['roi_scores'][index].reshape(-1)) 
            mask = (label_preds != 0).reshape(-1)

            box_preds = box_preds[mask, :]
            scores = scores[mask]
            labels = label_preds[mask]-1

            # currently don't need nms 
            pred_dict = {
                'box3d_lidar': box_preds,
                'scores': scores,
                'label_preds': labels,
                "metadata": batch_dict["metadata"][index]
            }

            pred_dicts.append(pred_dict)

        return pred_dicts 


    def forward(self, example, return_loss=True, **kwargs):
        out = self.single_det.forward_two_stage(example, 
            return_loss, **kwargs)

        if len(out) == 5:
            one_stage_pred, bev_feature, voxel_feature, final_feature, one_stage_loss = out 
            example['voxel_feature'] = voxel_feature
        elif len(out) == 3:
            one_stage_pred, bev_feature, one_stage_loss = out 
        else:
            raise NotImplementedError

        if self.roi_head.code_size == 7 and return_loss is True:
            # drop velocity 
            example['gt_boxes_and_cls'] = example['gt_boxes_and_cls'][:, :, [0, 1, 2, 3, 4, 5, 6, -1]]

        example = self.reorder_first_stage_pred_and_feature(first_pred=one_stage_pred, example=example)

        # final classification / regression 
        batch_dict = self.roi_head(example, training=return_loss)

        if return_loss:
            roi_loss, tb_dict = self.roi_head.get_loss()

            return self.combine_loss(one_stage_loss, roi_loss, tb_dict)
        else:
            return self.post_process(batch_dict)
