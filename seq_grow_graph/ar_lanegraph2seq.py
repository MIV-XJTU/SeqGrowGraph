# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------

import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.structures import InstanceData
import numpy as np

from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from mmdet3d.registry import MODELS
from mmdet3d.structures.ops import bbox3d2result
from .grid_mask import GridMask
from .LiftSplatShoot import LiftSplatShootEgo
from .core import seq2nodelist, seq2bznodelist, seq2plbznodelist, av2seq2bznodelist
from .decode_centerline import EvalMapBzGraphAAAI
from .bz_roadnet_reach_dist_eval import get_geom, get_range

@MODELS.register_module()
class AR_LG2Seq(MVXTwoStageDetector):
    """Petr3D. nan for all token except label"""
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
                 bz_grid_conf=None,
                 data_aug_conf=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 vis_cfg=None,
                 freeze_pretrain=True,
                 bev_scale=1.0,
                 epsilon=2,
                 max_vertex_num=50,
                 max_edge_len=500,
                 init_cfg=None,
                 data_preprocessor=None,
                 vis_dir="aaai"
                 ):
        super(AR_LG2Seq, self).__init__(pts_voxel_layer, pts_middle_encoder,
                                                        pts_fusion_layer, img_backbone, pts_backbone,
                                                        img_neck, pts_neck, pts_bbox_head, img_roi_head,
                                                        img_rpn_head, train_cfg, test_cfg, init_cfg,
                                                        data_preprocessor)
        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        # data_aug_conf = {
        #     'final_dim': (128, 352),
        #     'H': 900, 'W': 1600,
        # }
        # self.up = Up(512, 256, scale_factor=2)
        # view_transformers = []
        # view_transformers.append(
        #     LiftSplatShoot(grid_conf, data_aug_conf, downsample=16, d_in=256, d_out=256, return_bev=True))
        # self.view_transformers = nn.ModuleList(view_transformers)
        # self.view_transformers = LiftSplatShoot(grid_conf, data_aug_conf, downsample=16, d_in=256, d_out=256, return_bev=True)
        self.view_transformers = LiftSplatShootEgo(grid_conf, data_aug_conf, return_bev=True, **lss_cfg)
        self.downsample = lss_cfg['downsample']
        self.final_dim = data_aug_conf['final_dim']

        self.num_center_classes = 576
        self.box_range = 200
        self.coeff_range = 200
        self.vertex_id_start = 200
        self.connect_start = 250
        self.coeff_start = 300
        self.no_known = 575  # n/a and padding share the same label to be eliminated from loss calculation
        self.start = 574
        self.end_v = 573
        self.end_e = 572
        self.vis_cfg = vis_cfg
        self.bev_scale = bev_scale
        self.epsilon = epsilon
        self.max_vertex_num = max_vertex_num
        self.max_edge_len = max_edge_len

        self.grid_conf = grid_conf
        self.bz_grid_conf = bz_grid_conf
        self.vis_dir=vis_dir
        self.dx, bx, nx, self.pc_range, ego_points = get_geom(grid_conf)
        self.bz_dx, bz_bx, bz_nx, self.bz_pc_range = get_range(bz_grid_conf)
        
        if freeze_pretrain:
            self.freeze_pretrain()
    
    def freeze_pretrain(self):
        for m in self.img_backbone.parameters():
            m.requires_grad=False
        for m in self.img_neck.parameters():
            m.requires_grad=False
        for m in self.view_transformers.parameters():
            m.requires_grad=False

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
        img_feats = self.extract_img_feat(img, img_metas)
        largest_feat_shape = img_feats[0].shape[3]
        down_level = int(np.log2(self.downsample // (self.final_dim[0] // largest_feat_shape)))
        bev_feats = self.view_transformers(img_feats[down_level], img_metas)
        return bev_feats

    def forward_pts_train(self,
                          bev_feats,
                          gt_vert_sentences, 
                          gt_edge_sentences,
                          img_metas,
                          num_coeff, ):
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
        bs = len(gt_vert_sentences)
        device = bev_feats[0].device
        box_labels = []
        input_seqs = []
        output_seqs = []
        num_vertex = max([len(gt_vert_sentence) // 2 for gt_vert_sentence in gt_vert_sentences])
        max_vertex = max(num_vertex, self.max_vertex_num)  # 100
        edge_len = max([len(gt_edge_sentence) for gt_edge_sentence in gt_edge_sentences])
        max_edge_len = max(edge_len, self.max_edge_len)
        for bi in range(bs):
            vert = torch.tensor(gt_vert_sentences[bi], device=device).long()
            edge = torch.tensor(gt_edge_sentences[bi], device=device).long()

            random_vert = torch.rand(max_vertex - len(vert)//2, 2).to(device)
            random_vert = (random_vert * (self.box_range - 1)).long().flatten()
            random_edge = torch.rand(max_edge_len - len(edge)).to(device)
            random_edge = (random_edge * (self.coeff_range - 1)).long().flatten() + self.coeff_start
            in_vert_seq = torch.cat([torch.ones(1).to(vert) * self.start, vert, random_vert])
            in_edge_seq = torch.cat([torch.ones(1).to(edge) * self.end_v, edge, random_edge])

            na_vert = torch.ones(max_vertex*2 - len(vert)).to(device).long() * self.no_known
            na_edge = torch.ones(max_edge_len - len(edge)).to(device).long() * self.no_known
            out_vert_seq = torch.cat([vert, na_vert,torch.ones(1).to(vert) * self.end_v])
            out_edge_seq = torch.cat([edge, na_edge,torch.ones(1).to(edge) * self.end_e])

            input_seq = torch.cat([in_vert_seq, in_edge_seq])
            input_seqs.append(input_seq.unsqueeze(0))
            output_seq = torch.cat([out_vert_seq, out_edge_seq])
            output_seqs.append(output_seq.unsqueeze(0))
        input_seqs = torch.cat(input_seqs, dim=0)  # [8,501]
        output_seqs = torch.cat(output_seqs, dim=0)  # [8,501]
        outputs = self.pts_bbox_head(bev_feats, input_seqs, img_metas)[-1]


        # =============================================
        # 训练可视化
        
   
        # for bi in range(outputs.shape[0]):
        #     try:
        #         pred_line_seq = outputs[bi]
        #         pred_line_seq = pred_line_seq.argmax(-1)
        #         if self.end_e in pred_line_seq:
        #             stop_idx = (pred_line_seq == self.end_e).nonzero(as_tuple=True)[0][0]
        #         else:
        #             stop_idx = len(pred_line_seq)
         
        #         pred_line_seq = pred_line_seq[:stop_idx]
        #         pred_graph = EvalMapBzGraphAAAI(img_metas[bi]['token'],pred_line_seq.detach().cpu().numpy().tolist(),pc_range=self.pc_range,dx=self.dx,bz_pc_range=self.bz_pc_range,bz_dx=self.bz_dx)
        #         pred_graph.visualization([200, 200], os.path.join(self.vis_dir,'train'), 'n', 'n')
        #     except:
        #         import traceback
        #         traceback.print_exc()
        #     break

        # =============================================
 

        gt_seq = output_seqs[output_seqs != self.no_known].long()
        
        preds_dicts = dict(
            pred_seq=outputs[output_seqs != self.no_known],
        )

        loss_inputs = [gt_seq, preds_dicts]
        losses = self.pts_bbox_head.loss_by_feat(*loss_inputs)

        return losses
    
    def loss(self,
             inputs=None,
             data_samples=None,**kwargs):

        img = inputs['img']
        img_metas = [ds.metainfo for ds in data_samples]

        bev_feats = self.extract_feat(img=img, img_metas=img_metas)
        if self.bev_scale != 1.0:
            b, c, h, w = bev_feats.shape
            bev_feats = F.interpolate(bev_feats, (int(h * self.bev_scale), int(w * self.bev_scale)))
        losses = dict()
        gt_vert_sentence = [img_meta['vert_sentence'] for img_meta in img_metas]
        gt_edge_sentence = [img_meta['edge_sentence'] for img_meta in img_metas]
        n_control = img_metas[0]['n_control']
        num_coeff = n_control - 2
        losses_pts = self.forward_pts_train(bev_feats, gt_vert_sentence, gt_edge_sentence, img_metas, num_coeff)
        losses.update(losses_pts)
        return losses
        

    def forward_test(self, img_metas, img=None, **kwargs):
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        img = [img] if img is None else img
        return self.simple_test(img_metas , img, **kwargs)

    def vis_linepts(self, pred_points, pred_labels, img_meta, path, aux_name):
        import cv2

        label_color = [[255, 0, 0], [0, 0, 230], [35, 244, 232], [70, 70, 70], [102, 102, 156],
               [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
               [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
               [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
               [0, 80, 100],  [119, 11, 32]]
        bev_img = np.zeros((200, 200, 3), np.uint8)
        for i in range(len(pred_points)):
            try:
                bev_img = cv2.circle(bev_img, (int(pred_points[i][0]), int(pred_points[i][1])), 1, label_color[int(pred_labels[i])], 2)
            except:
                pass

        name = img_meta['filename'][0].split('/')[-1].split('.jpg')[0]
        save_dir = f"vis/{path}/"
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        cv2.imwrite(os.path.join(save_dir, f"{name}_{aux_name}.png"), bev_img)
        return
    
    def vis_from_nodelist(self, nodelist, img_meta, path, aux_name):
        Map_size = [(-50, 50), (-50, 50)]
        Map_resolution = 0.5
        image = np.zeros([200, 200, 3])
        point_color_map = {"start": (0, 0, 255), 'fork': (0, 255, 0), "continue": (0, 255, 255), "merge": (255, 0, 0)}

        for idx, node in enumerate(nodelist):
            if node['sque_type'] == 'start':
                cv2.circle(image, node['coord'], 1, color=point_color_map['start'], thickness=2)
            elif node['sque_type'] == 'continue':
                cv2.circle(image, node['coord'], 1, color=point_color_map['continue'], thickness=2)
                # cv2.polylines(image, [subgraphs_points_in_between_nodes[(node.sque_index-1, node.sque_index)]], False, color=point_color_map['continue'], thickness=1)
                cv2.arrowedLine(image, nodelist[idx - 1]['coord'], node['coord'], color=point_color_map['continue'],
                                thickness=1, tipLength=0.1)
            elif node['sque_type'] == 'fork':
                if node['fork_from'] > idx or node['fork_from'] < 0:
                    continue
                cv2.circle(image, node['coord'], 1, color=point_color_map['fork'], thickness=2)
                cv2.arrowedLine(image, nodelist[node['fork_from'] - 1]['coord'], node['coord'],
                                color=point_color_map['fork'],
                                thickness=1, tipLength=0.1)
            elif node['sque_type'] == 'merge':
                if node['merge_with'] > idx or node['merge_with'] < 0:
                    continue
                cv2.circle(image, node['coord'], 1, color=point_color_map['merge'], thickness=2)
                cv2.arrowedLine(image, nodelist[node['merge_with'] - 1]['coord'], node['coord'],
                                color=point_color_map['merge'], thickness=1, tipLength=0.1)

        name = img_meta['filename'][0].split('/')[-1].split('.jpg')[0]
        save_dir = f"vis/{path}/"
        os.makedirs(save_dir, exist_ok=True)
        # if not os.path.exists(save_dir):
        #     os.mkdir(save_dir)
        cv2.imwrite(os.path.join(save_dir, f"{name}_{aux_name}.png"), image)

    def simple_test_pts(self, pts_feats, img_metas, rescale=False):
        """Test function of point cloud branch."""
        # gt_lines_seqs = [img_meta['centerline_sequence'] for img_meta in img_metas]
        n_control = img_metas[0]['n_control']
        num_coeff = n_control - 2
        clause_length = 4 + num_coeff * 2
        # gt_node_list = self.seq2nodelist(gt_lines_seqs[0])
        # self.vis_from_nodelist(gt_node_list, img_metas[0], self.vis_cfg['path'], 'gt')
        device = pts_feats[0].device
        input_seqs = (torch.ones(pts_feats.shape[0], 1).to(device) * self.start).long()
        outs = self.pts_bbox_head(pts_feats, input_seqs, img_metas)
        output_seqs, values = outs
        line_results = []
        for bi in range(output_seqs.shape[0]):
            # gt_node_list = self.seq2nodelist(gt_lines_seqs[bi])
            pred_line_seq = output_seqs[bi]
            pred_line_seq = pred_line_seq[1:]
            # if self.end in pred_line_seq:

            if self.end_e in pred_line_seq:
                stop_idx = (pred_line_seq == self.end_e).nonzero(as_tuple=True)[0][0]
            else:
                stop_idx = len(pred_line_seq)
            
            
            pred_line_seq = pred_line_seq[:stop_idx]

            line_results.append(dict(
                line_seqs = pred_line_seq.detach().cpu().numpy(),
     
            ))
            

        return line_results

    def simple_test(self, img_metas, img=None, rescale=False):
        """Test function without augmentaiton."""
        bev_feats = self.extract_feat(img=img, img_metas=img_metas)

        bbox_list = [dict() for i in range(len(img_metas))]
        line_results = self.simple_test_pts(
            bev_feats, img_metas, rescale=rescale)
        for result_dict, line_result, img_meta in zip(bbox_list, line_results, img_metas):
            result_dict['line_results'] = line_result
            result_dict['token'] = img_meta['token']
            try:
                pred_graph = EvalMapBzGraphAAAI(img_meta['token'], line_result["line_seqs"] ,pc_range=self.pc_range,dx=self.dx,bz_pc_range=self.bz_pc_range,bz_dx=self.bz_dx)
                pred_graph.visualization([200, 200], os.path.join(self.vis_dir,'test'), 'n', 'n')
            except:
                import traceback
                traceback.print_exc()
            
            
        return bbox_list

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
        img_metas = [ds.metainfo for ds in batch_data_samples]

        seg_list = self.simple_test(img_metas, img)

        

        return seg_list

