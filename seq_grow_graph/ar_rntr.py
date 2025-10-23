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
from .decode_centerline import EvalMapBzGraph
from .encode_centerline import convert_coeff_coord
from .bz_roadnet_reach_dist_eval import get_geom, get_range


@MODELS.register_module()
class AR_RNTR(MVXTwoStageDetector):
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
                 max_box_num=100,
                 init_cfg=None,
                 data_preprocessor=None,front_camera_only=False,vis_dir="original",
                 ):
        super(AR_RNTR, self).__init__(pts_voxel_layer, pts_middle_encoder,
                                                        pts_fusion_layer, img_backbone, pts_backbone,
                                                        img_neck, pts_neck, pts_bbox_head, img_roi_head,
                                                        img_rpn_head, train_cfg, test_cfg, init_cfg,
                                                        data_preprocessor)
        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.front_camera_only=front_camera_only
        self.vis_dir=vis_dir

        self.view_transformers = LiftSplatShootEgo(grid_conf, data_aug_conf, return_bev=True, **lss_cfg)
        self.downsample = lss_cfg['downsample']
        self.final_dim = data_aug_conf['final_dim']

        self.num_center_classes = 576
        self.box_range = 200
        self.coeff_range = 200
        self.num_classes=4
        self.category_start = 200
        self.connect_start = 250
        self.coeff_start = 350
        self.no_known = 575  # n/a and padding share the same label to be eliminated from loss calculation
        self.start = 574
        self.end = 573
        self.noise_connect = 572
        self.noise_label = 571
        self.noise_coeff = 570
        self.vis_cfg = vis_cfg
        self.bev_scale = bev_scale
        self.epsilon = epsilon
        self.max_box_num = max_box_num

        self.grid_conf = grid_conf
        self.bz_grid_conf = bz_grid_conf

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
                          gt_lines_coords,
                          gt_lines_labels,
                          gt_lines_connects,
                          gt_lines_coeffs,
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
        device = bev_feats[0].device
        box_labels = []
        input_seqs = []
        output_seqs = []
        max_box = max([len(target) for target in gt_lines_coords])
        num_box = max(max_box + 2, self.max_box_num)  # 100
        coeff_dim = num_coeff * 2
        for bi in range(len(gt_lines_coords)):
            box = torch.tensor(gt_lines_coords[bi], device=device).long()
            box = box.reshape(-1,2)
            label = torch.tensor(gt_lines_labels[bi], device=device).long() + self.category_start  # [8,1]
            label = label.reshape(-1,1)
            connect = torch.tensor(gt_lines_connects[bi], device=device).long() + self.connect_start  # [8,1]
            connect = connect.reshape(-1,1)
            coeff = torch.tensor(gt_lines_coeffs[bi], device=device).long() + self.coeff_start  # [8,1]
            coeff = coeff.reshape(-1, coeff_dim)
            box_label = torch.cat([box, label, connect, coeff], dim=-1)  # [8, 5]

            random_box = torch.rand(num_box - box_label.shape[0], 2).to(device)
            random_box = (random_box * (self.box_range - 1)).int()
            random_label = torch.randint(0, self.num_classes, (num_box - box_label.shape[0], 1)).to(label)
            random_label = random_label + self.category_start
            random_connect = torch.randint(0, num_box, (num_box - box_label.shape[0], 1)).to(label)
            random_connect = random_connect + self.connect_start
            random_coeff = torch.rand(num_box - box_label.shape[0], coeff_dim).to(device)
            random_coeff = (random_coeff * (self.coeff_range - 1)).int()
            random_coeff = random_coeff + self.coeff_start
            random_box_label = torch.cat([random_box, random_label.int(), random_connect.int(), random_coeff.int()], dim=-1)  # [92, 5]

            input_seq = torch.cat([box_label, random_box_label], dim=0)  # [100, 5]

            input_seq = torch.cat([torch.ones(1).to(box_label) * self.start, input_seq.flatten()])  # [501]
            input_seqs.append(input_seq.unsqueeze(0))

            output_na = torch.ones(num_box - box_label.shape[0], 1).to(input_seq) * self.no_known  # [92,3]
            output_noise = torch.ones(num_box - box_label.shape[0], 1).to(input_seq) * self.no_known   # [92,1]
            output_noise_connect = torch.ones(num_box - box_label.shape[0], 1).to(input_seq) * self.no_known  # [92,1]
            output_noise_coeff = torch.ones(num_box - box_label.shape[0], coeff_dim).to(input_seq) * self.no_known  # [92,1]
            output_end = torch.ones(num_box - box_label.shape[0], 1).to(input_seq) * self.no_known # [92, 1]
            output_seq = torch.cat([output_na, output_noise, output_noise_connect, output_noise_coeff, output_end], dim=-1)  # [92,5]

            output_seq = torch.cat([box_label.flatten(), torch.ones(1).to(box_label) * self.end, output_seq.flatten()])
            output_seqs.append(output_seq.unsqueeze(0))
        input_seqs = torch.cat(input_seqs, dim=0)  # [8,501]
        output_seqs = torch.cat(output_seqs, dim=0)  # [8,501]
        box_labels = output_seqs[:, :-1].flatten()  # [4008]

        outputs = self.pts_bbox_head(bev_feats, input_seqs, img_metas)[-1, :, :-1, :]

        clause_length = 4 + coeff_dim
        n_control = img_metas[0]['n_control']
        
        
        # =============================================
        # 训练可视化
        
        # line_results = []
        # for bi in range(outputs.shape[0]):
        #     pred_line_seq = outputs[bi]
        #     pred_line_seq = pred_line_seq.argmax(-1)
        #     if self.end in pred_line_seq:
        #         stop_idx = (pred_line_seq == self.end).nonzero(as_tuple=True)[0][0]
        #     else:
        #         stop_idx = len(pred_line_seq)
        #     stop_idx = stop_idx // clause_length * clause_length
        #     pred_line_seq = pred_line_seq[:stop_idx]
        #     pred_line_seq[2::clause_length] = pred_line_seq[2::clause_length] - self.category_start
        #     pred_line_seq[3::clause_length] = pred_line_seq[3::clause_length] - self.connect_start
        #     for k in range(4, clause_length):
        #         pred_line_seq[k::clause_length] = pred_line_seq[k::clause_length] - self.coeff_start
        #     pred_node_list = av2seq2bznodelist(pred_line_seq.detach().cpu().numpy(), n_control, self.epsilon)

        #     line_results.append(dict(
        #         line_seqs = pred_line_seq.detach().cpu(),
        #         pred_node_lists = pred_node_list
        #     ))
        #     break
      
        # bbox_list = [dict() for i in range(len(img_metas))]
        # for result_dict, line_result, img_meta in zip(bbox_list, line_results, img_metas):
        #     try:
        #         result_dict['line_results'] = line_result
        #         result_dict['token'] = img_meta['token']

        #         pred_node_lists = convert_coeff_coord(line_result['pred_node_lists'], self.pc_range, self.dx, self.bz_pc_range, self.bz_dx)

        #         pred_graph = EvalMapBzGraph(img_meta['token'], pred_node_lists)
        #         pred_graph.visualization([200, 200], os.path.join(self.vis_dir,'train'), 'n', 'n')
        #     except:
        #         import traceback
        #         traceback.print_exc()
            
        # =============================================
        
        outputs = outputs.reshape(-1, self.num_center_classes)  # [602, 2003] last layer
        outputs_pos = torch.cat([outputs[::clause_length, :], outputs[1::clause_length, :]])
        outputs_cls = outputs[2::clause_length, :]
        outputs_connects = outputs[3::clause_length, :]
        outputs_coeffs = torch.cat([outputs[k::clause_length, :] for k in range(4, clause_length)])
        inputs_pos = torch.cat([box_labels[::clause_length], box_labels[1::clause_length]])
        inputs_cls = box_labels[2::clause_length]
        inputs_connects = box_labels[3::clause_length]
        inputs_coeffs = torch.cat([box_labels[k::clause_length] for k in range(4, clause_length)])

        gt_coords = [inputs_pos[inputs_pos != self.no_known].long()]
        gt_labels = [inputs_cls[inputs_cls != self.no_known].long()]
        gt_connects = [inputs_connects[inputs_connects != self.no_known].long()]
        gt_coeffs = [inputs_coeffs[inputs_coeffs != self.no_known].long()]
        
        preds_dicts = dict(
            preds_coords=[outputs_pos[inputs_pos != self.no_known]],
            preds_labels=[outputs_cls[inputs_cls != self.no_known]],
            preds_connects=[outputs_connects[inputs_connects != self.no_known]],
            preds_coeffs=[outputs_coeffs[inputs_coeffs != self.no_known]]
        )

        loss_inputs = [gt_coords, gt_labels, gt_connects, gt_coeffs, preds_dicts]
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
        gt_lines_coords = [img_meta['centerline_coord'] for img_meta in img_metas]
        gt_lines_labels = [img_meta['centerline_label'] for img_meta in img_metas]
        gt_lines_connects = [img_meta['centerline_connect'] for img_meta in img_metas]
        gt_lines_coeffs = [img_meta['centerline_coeff'] for img_meta in img_metas]
        n_control = img_metas[0]['n_control']
        num_coeff = n_control - 2
        losses_pts = self.forward_pts_train(bev_feats, gt_lines_coords, gt_lines_labels, 
                                            gt_lines_connects, gt_lines_coeffs,
                                            img_metas, num_coeff)
        losses.update(losses_pts)
        return losses
    
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
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        batch_input_imgs = batch_inputs_dict['img']
        return self.simple_test(batch_input_metas, batch_input_imgs)

    def simple_test_pts(self, pts_feats, img_metas):
        """Test function of point cloud branch."""
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
            pred_line_seq = output_seqs[bi]
            pred_line_seq = pred_line_seq[1:]
            if self.end in pred_line_seq:
                stop_idx = (pred_line_seq == self.end).nonzero(as_tuple=True)[0][0]
            else:
                stop_idx = len(pred_line_seq)
            stop_idx = stop_idx // clause_length * clause_length
            pred_line_seq = pred_line_seq[:stop_idx]
            pred_line_seq[2::clause_length] = pred_line_seq[2::clause_length] - self.category_start
            pred_line_seq[3::clause_length] = pred_line_seq[3::clause_length] - self.connect_start
            for k in range(4, clause_length):
                pred_line_seq[k::clause_length] = pred_line_seq[k::clause_length] - self.coeff_start
            # pred_line_seq=pred_line_seq.repeat(2)
            pred_node_list = av2seq2bznodelist(pred_line_seq.detach().cpu().numpy(), n_control, self.epsilon)

            line_results.append(dict(
                line_seqs = pred_line_seq.detach().cpu(),
                pred_node_lists = pred_node_list
            ))
        return line_results

    def simple_test(self, img_metas, img=None):
        """Test function without augmentaiton."""
        bev_feats = self.extract_feat(img=img, img_metas=img_metas)
        bbox_list = [dict() for i in range(len(img_metas))]
        line_results = self.simple_test_pts(
            bev_feats, img_metas)
        for result_dict, line_result, img_meta in zip(bbox_list, line_results, img_metas):
            result_dict['line_results'] = line_result
            result_dict['token'] = img_meta['token']
            try:
                pred_node_lists = convert_coeff_coord(line_result['pred_node_lists'], self.pc_range, self.dx, self.bz_pc_range, self.bz_dx)

                pred_graph = EvalMapBzGraph(img_meta['token'], pred_node_lists,front_camera_only=self.front_camera_only)
                pred_graph.visualization([200, 200], os.path.join(self.vis_dir,'test'), 'n', 'n')
            except:
                import traceback
                traceback.print_exc()
        return bbox_list

