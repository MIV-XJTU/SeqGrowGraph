
import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.structures import InstanceData
import numpy as np

from mmcv.cnn import ConvModule
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from mmdet3d.registry import MODELS
from mmdet3d.structures.ops import bbox3d2result
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.utils import resize
from mmseg.models.losses import accuracy
from .grid_mask import GridMask
from .LiftSplatShoot import LiftSplatShootEgo
from .core import seq2nodelist, seq2bznodelist, seq2plbznodelist, av2seq2bznodelist


@MODELS.register_module()
class RoadSegHead(BaseDecodeHead):
    def __init__(self, 
                 num_convs=2,
                 kernel_size=3,
                 concat_input=True,
                 dilation=1,
                 upsample=5,
                 **kwargs):
        if 'train_cfg' in kwargs:
            del kwargs['train_cfg']
        if 'test_cfg' in kwargs:
            del kwargs['test_cfg']
        assert num_convs >= 0 and dilation > 0 and isinstance(dilation, int)
        self.num_convs = num_convs
        self.concat_input = concat_input
        self.kernel_size = kernel_size
        self.upsample = upsample
        super(RoadSegHead, self).__init__(**kwargs)
        if num_convs == 0:
            assert self.in_channels == self.channels

        conv_padding = (kernel_size // 2) * dilation
        convs = []
        convs.append(
            ConvModule(
                self.in_channels,
                self.channels,
                kernel_size=kernel_size,
                padding=conv_padding,
                dilation=dilation,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))
        for i in range(num_convs - 1):
            convs.append(
                ConvModule(
                    self.channels,
                    self.channels,
                    kernel_size=kernel_size,
                    padding=conv_padding,
                    dilation=dilation,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
        if num_convs == 0:
            self.convs = nn.Identity()
        else:
            self.convs = nn.ModuleList(convs)
        if self.concat_input:
            self.conv_cat = ConvModule(
                self.in_channels + self.channels,
                self.channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
    
    def forward(self, bev_feats):
        output = bev_feats
        ih, iw = bev_feats.shape[2:]
        for conv in self.convs:
            output = conv(output)
        if self.concat_input:
            output = self.conv_cat(torch.cat([bev_feats, output], dim=1))
        output = self.cls_seg(output)
        output = F.interpolate(output, (ih*self.upsample, iw*self.upsample), mode='bilinear', align_corners=self.align_corners)
        return output
    
    def loss_by_feat(self, seg_logits,
                     seg_label) -> dict:
        """Compute segmentation loss.

        Args:
            seg_logits (Tensor): The output from decode head forward function.
            batch_data_samples (List[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        loss = dict()
        seg_logits = resize(
            input=seg_logits,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logits, seg_label)
        else:
            seg_weight = None
        seg_label = seg_label.squeeze(1)

        loss['ce_loss'] = self.loss_decode(
            seg_logits,
            seg_label,
            weight=seg_weight,
            ignore_index=self.ignore_index)

        loss['acc_seg'] = accuracy(
            seg_logits, seg_label, ignore_index=self.ignore_index)
        return loss


@MODELS.register_module()
class RoadSeg(MVXTwoStageDetector):
    """Petr3D."""
    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 lss_cfg=None,
                 grid_conf=None,
                 data_aug_conf=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 vis_cfg=None,
                 init_cfg=None,
                 data_preprocessor=None,
                 ):
        super(RoadSeg, self).__init__(pts_voxel_layer, pts_middle_encoder,
                                                        pts_fusion_layer, img_backbone, pts_backbone,
                                                        img_neck, pts_neck, pts_bbox_head, img_roi_head,
                                                        img_rpn_head, train_cfg, test_cfg, init_cfg,
                                                        data_preprocessor)
        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.view_transformers = LiftSplatShootEgo(grid_conf, data_aug_conf, return_bev=True, **lss_cfg)
        self.downsample = lss_cfg['downsample']
        self.final_dim = data_aug_conf['final_dim']

        self.vis_cfg = vis_cfg

    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        # print(img[0].size())
        if isinstance(img, list):
            img = torch.stack(img, dim=0)

        B = img.size(0)

        if img is not None:
            input_shape = img.shape[-2:]
            # update real input shape of each single img
            for img_meta in img_metas:
                img_meta.update(input_shape=input_shape)
            if img.dim() == 5:
                if img.size(0) == 1 and img.size(1) != 1:
                    img.squeeze_()
                else:
                    B, N, C, H, W = img.size()
                    img = img.view(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)
            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    def extract_feat(self, img, img_metas):
        """Extract features from images and points."""
        # if isinstance(img, dict):
        #     img = img['img']
        img_feats = self.extract_img_feat(img, img_metas)
        largest_feat_shape = img_feats[0].shape[3]
        down_level = int(np.log2(self.downsample // (self.final_dim[0] // largest_feat_shape)))
        bev_feats = self.view_transformers(img_feats[down_level], img_metas)
        return bev_feats
    
    def loss(self,
             inputs=None,
             data_samples=None,**kwargs):

        img = inputs['img']
        center_seg = inputs['center_seg']
        img_metas = [ds.metainfo for ds in data_samples]
        bev_feats = self.extract_feat(img=img, img_metas=img_metas)
        losses = dict()
        losses_pts = self.forward_pts_train(bev_feats, center_seg)
        losses.update(losses_pts)
        return losses

    def forward_pts_train(self,
                          bev_feats,
                          center_seg,
                          img_metas=None,
                          ):
        """Forward function for point cloud branch.
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
        Returns:
            dict: Losses of each branch.
        """
        if isinstance(center_seg, list):
            center_seg = torch.stack(center_seg, dim=0)
        loss_inputs = [bev_feats, center_seg.unsqueeze(1), None]
        losses = self.pts_bbox_head.loss(*loss_inputs)
        return losses

    def forward_test(self, img_metas, img=None, **kwargs):
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        # return self.simple_test(img_metas[0], img[0], **kwargs)
        import pdb
        pdb.set_trace()
        print('need debug')
        return self.simple_test(img_metas, img, **kwargs)

    def simple_test_pts(self, bev_feats, img_metas, rescale=False):
        """Test function of point cloud branch."""
        outputs = self.pts_bbox_head(bev_feats)
        seg_pred = outputs.argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        # self.vis_centerline(seg_pred[0], img_metas[0], 'centerline_d8_val_20')
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred

    def simple_test(self, img_metas, img=None, rescale=False):
        """Test function without augmentaiton."""
        bev_feats = self.extract_feat(img=img, img_metas=img_metas)
        seg_list = [dict(gt_seg_maps=img_meta['sample_idx']) for img_meta in img_metas]
        seg_preds = self.simple_test_pts(bev_feats, img_metas)
        for result_dict, seg_pred in zip(seg_list, seg_preds):
            result_dict['road_seg'] = seg_pred
        return seg_list

    def aug_test_pts(self, feats, img_metas, rescale=False):
        feats_list = []
        for j in range(len(feats[0])):
            feats_list_level = []
            for i in range(len(feats)):
                feats_list_level.append(feats[i][j])
            feats_list.append(torch.stack(feats_list_level, -1).mean(-1))
        outs = self.pts_bbox_head(feats_list, img_metas)
        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def aug_test(self, img_metas, imgs=None, rescale=False):
        """Test function with augmentaiton."""
        img_feats = self.extract_feats(img_metas, imgs)
        img_metas = img_metas[0]
        bbox_list = [dict() for i in range(len(img_metas))]
        bbox_pts = self.aug_test_pts(img_feats, img_metas, rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return bbox_list

    def predict(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Forward of testing.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                'points' keys.

                - points (list[torch.Tensor]): Point cloud of each sample.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`.

        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input sample. Each Det3DDataSample usually contain
            'pred_instances_3d'. And the ``pred_instances_3d`` usually
            contains following keys.

            - scores_3d (Tensor): Classification scores, has a shape
                (num_instances, )
            - labels_3d (Tensor): Labels of bboxes, has a shape
                (num_instances, ).
            - bbox_3d (:obj:`BaseInstance3DBoxes`): Prediction of bboxes,
                contains a tensor with shape (num_instances, 7).
        """

        img = batch_inputs_dict['img']
        center_seg = batch_inputs_dict['center_seg']
        img_metas = [ds.metainfo for ds in batch_data_samples]

        seg_list = self.simple_test(img_metas, img)

        for gt_seg, pred_seg_dict in zip(center_seg, seg_list):
            gt_seg = gt_seg.cpu().numpy()
            gt_seg_maps = pred_seg_dict['gt_seg_maps']
            pred_seg = pred_seg_dict['road_seg']
            pred_seg_dict['gt_seg'] = gt_seg

            gt_image = np.array(gt_seg * 255, dtype=np.uint8)
            pred_image = np.array(pred_seg * 255, dtype=np.uint8)

            pred_image_resize = cv2.resize(pred_image, gt_image.shape[::-1], interpolation=cv2.INTER_AREA)

            final_img = np.concatenate([gt_image, pred_image_resize], axis=-1)
            os.makedirs('./vis_roadseg', exist_ok=True)
            cv2.imwrite(f'./vis_roadseg/{gt_seg_maps}.jpg', final_img)

        return seg_list
