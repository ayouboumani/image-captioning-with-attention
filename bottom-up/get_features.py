import _init_paths
import os
import numpy as np
import cv2
import torch
import time
import argparse
from tqdm import tqdm
from model.utils.config import cfg, cfg_from_file
from model.faster_rcnn.resnet import resnet
from utils_ import get_image_blob, save_features
from numpy_nms.cpu_nms import cpu_nms



def get_features():
    # Load arguments.
    MIN_BOXES = 36
    MAX_BOXES = 36
    N_CLASSES = 1601
    CONF_THRESH = 0.2
    cfg_file='cfgs/faster_rcnn_resnet101.yml'
    model_file='models/bottomup_pretrained_10_100.pth'
    image_dir='../images'


    use_cuda = torch.cuda.is_available()
    assert use_cuda, 'Works only with CUDA'
    device = torch.device('cuda') if use_cuda else torch.device('cpu')
    cfg.CUDA = use_cuda
    np.random.seed(cfg.RNG_SEED)

    # Load the model.
    fasterRCNN = resnet(N_CLASSES, 101, pretrained=False)
    fasterRCNN.create_architecture()
    fasterRCNN.load_state_dict(torch.load(model_file))
    fasterRCNN.to(device)
    fasterRCNN.eval()
    print('Model is loaded.')

    # Load images.
    imglist = os.listdir(image_dir)
    num_images = len(imglist)
    print('Number of images: {}.'.format(num_images))
    features=[]
    images=[]
    # Extract features.
    for im_file in imglist:
        im = cv2.imread(os.path.join(image_dir, im_file))
        images.append(im)
        blobs, im_scales = get_image_blob(im)
        assert len(im_scales) == 1, 'Only single-image batch is implemented'
    
        im_data = torch.from_numpy(blobs).permute(0, 3, 1, 2).to(device)
        im_info = torch.tensor([[blobs.shape[1], blobs.shape[2], im_scales[0]]]).to(device)
        gt_boxes = torch.zeros(1, 1, 5).to(device)
        num_boxes = torch.zeros(1).to(device)
    
        with torch.set_grad_enabled(False):
            rois, cls_prob, _, _, _, _, _, _, \
            pooled_feat = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)
    
        boxes = rois.data.cpu().numpy()[:, :, 1:5].squeeze()
        boxes /= im_scales[0]
        cls_prob = cls_prob.data.cpu().numpy().squeeze()
        pooled_feat = pooled_feat.data.cpu().numpy()
    
        # Keep only the best detections.
        max_conf = np.zeros((boxes.shape[0]))
        for cls_ind in range(1, cls_prob.shape[1]):
            cls_scores = cls_prob[:, cls_ind]
            dets = np.hstack((boxes, cls_scores[:, np.newaxis])).astype(np.float32)
            keep = np.array(cpu_nms(dets, cfg.TEST.NMS))
            max_conf[keep] = np.where(cls_scores[keep] > max_conf[keep], cls_scores[keep], max_conf[keep])
    
        keep_boxes = np.where(max_conf >= CONF_THRESH)[0]
        if len(keep_boxes) < MIN_BOXES:
            keep_boxes = np.argsort(max_conf)[::-1][:MIN_BOXES]
        elif len(keep_boxes) > MAX_BOXES:
            keep_boxes = np.argsort(max_conf)[::-1][:MAX_BOXES]
    
        image_feat = pooled_feat[keep_boxes]
        features.append(image_feat)
    return images,features
