import argparse
import os, sys
import shutil
import time
from pathlib import Path
import imageio

# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append("/home/ubuntu/workspace/ywshin/construct/YOLOP/")
import time
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import scipy.special
import numpy as np
import torchvision.transforms as transforms
import PIL.Image as image
from tqdm import tqdm

from lib.config import cfg
from lib.config import update_config
from lib.utils.utils import create_logger, select_device, time_synchronized
from lib.models import get_net
from lib.dataset import LoadImages, LoadStreams
from lib.core.general import non_max_suppression, scale_coords
from lib.utils import plot_one_box,show_seg_result
# from lib.core.function import AverageMeter
# from lib.core.postprocess import morphological_process, connect_lane

from strong_sort.utils.parser import get_config
from strong_sort.strong_sort import StrongSORT

normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y 

def calc_distance(pt_1, pt_2):
    pt_1 = np.array((pt_1[0], pt_1[1]))
    pt_2 = np.array((pt_2[0], pt_2[1]))
    return np.linalg.norm(pt_1-pt_2)

def calc_center(pt):
    x_cent = (pt[0]+pt[2])/2
    y_cent = (pt[1]+pt[3])/2
    return [x_cent, y_cent]

class Patrasche:
    def __init__(self, cfg, opt):
        self.cfg = cfg
        self.opt = opt
        
        device = select_device(None,opt.device)
        if os.path.exists(opt.save_dir):  # output dir
            shutil.rmtree(opt.save_dir)  # delete dir
        os.makedirs(opt.save_dir)

        half = device.type != 'cpu'

        self.model = get_net(cfg)
        checkpoint = torch.load(opt.weights, map_location= device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model = self.model.to(device)
        
        if half:
            self.model.half()
        
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.names))]
        
        self.vid_path, self.vid_writer = None, None
        img = torch.zeros((1, 3, opt.img_size, opt.img_size), device=device)  # init img
        _ = self.model(img.half() if half else img) if device.type != 'cpu' else None  # run once
        self.model.eval()
        
        self.half = half
        self.device = device
        
        self.load_tracker(opt.strong_sort_weights)
    
    
    def load_tracker(self, strong_sort_weights):
        self.strongsort = StrongSORT(
                strong_sort_weights,
                self.device,
                max_dist=self.cfg.STRONGSORT.MAX_DIST,
                max_iou_distance=self.cfg.STRONGSORT.MAX_IOU_DISTANCE,
                max_age=self.cfg.STRONGSORT.MAX_AGE,
                n_init=self.cfg.STRONGSORT.N_INIT,
                nn_budget=self.cfg.STRONGSORT.NN_BUDGET,
                mc_lambda=self.cfg.STRONGSORT.MC_LAMBDA,
                ema_alpha=self.cfg.STRONGSORT.EMA_ALPHA,
        )
        #         return self.strongsort
    
    def set_dataloader(self):
        if self.opt.source.isnumeric():
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(self.opt.source, img_size=self.opt.img_size)
            bs = len(dataset)  # batch_size
        else:
            dataset = LoadImages(self.opt.source, img_size=self.opt.img_size)
            bs = 1  # batch_size
            
        return dataset
    
    def process_segmentation(self, img_det, da_seg_out, shapes, width, height):
        pad_w, pad_h = shapes[1][1]
        pad_w = int(pad_w)
        pad_h = int(pad_h)
        ratio = shapes[1][0][1]
        
        da_predict = da_seg_out[:, :, pad_h:(height-pad_h),pad_w:(width-pad_w)]
        da_seg_mask = torch.nn.functional.interpolate(da_predict, scale_factor=int(1/ratio), mode='bilinear')
        _, da_seg_mask = torch.max(da_seg_mask, 1)
        da_seg_mask = da_seg_mask.int().squeeze().cpu().numpy()

        center_w, center_h = da_seg_mask.shape[1]/2, da_seg_mask.shape[0]/2

        color_area = np.zeros((da_seg_mask.shape[0], da_seg_mask.shape[1], 1), dtype=np.uint8)
        color_area[da_seg_mask == 1] = 255
        color_seg = color_area[..., ::-1]

        img_det = cv2.resize(img_det, (da_seg_mask.shape[1], da_seg_mask.shape[0]))
        blank_img = np.zeros([da_seg_mask.shape[0], da_seg_mask.shape[1], 1], dtype = np.uint8)

        if self.opt.seg_visualization:
            ori_img_det = img_det.copy()
            ori_img_det[da_seg_mask == 1] = [0,255,0]
            alpha = 0.7
            img_det = cv2.addWeighted(img_det, alpha, ori_img_det, 1-alpha, 1.0)
#             img_det = show_seg_result(img_det, da_seg_mask, _, _, is_demo=True)

    
        return img_det, blank_img, color_seg, center_w, center_h
    
    
    def process_detection(self, det, img, ori_img, img_det, color_seg, blank_img, center_w, center_h):
        person_list = []
        conf_list = []
        cls_list = []
        obs_list = []
        
        if len(det):
            det[:,:4] = scale_coords(img.shape[2:],det[:,:4],img_det.shape).round()
            for *xyxy,conf,cls in reversed(det):
                if self.names[int(cls)] == "person":
                    person_list.append(xyxy)
                    conf_list.append(conf)
                    cls_list.append(cls)
#                 elif self.opt.track_visualization: #만약 영상을 사람 1명만 있는 것으로 실행시킬 계획이라면
                if self.opt.track_visualization: #만약 사람이 여럿 있는 영상 실행 계획이라면 (obstacle과 겹쳐보일수도)
                    label_det_pred = f'{"obstacle"} {conf:.2f}'
                    plot_one_box(xyxy, img_det, label=label_det_pred, color=self.colors[int(cls)], line_thickness=2)

                obs_list.append(xyxy)
        
        return self.process_tracking(ori_img, img_det, color_seg, blank_img, person_list, conf_list, cls_list, obs_list, center_w, center_h)

    
    def process_tracking(self, ori_img, img_det, color_seg, blank_img, person_list, conf_list, cls_list, obs_list, center_w, center_h):
        road_check = 0
        result = "STOP"
        
        if len(person_list):  # detections per image
            xywhs = xyxy2xywh(torch.Tensor(person_list))
            confs = torch.tensor(conf_list)
            clss = torch.tensor(cls_list)
            
#             t3 = time.time()
            outputs = self.strongsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), ori_img)
#             print(time.time() - t3)
            
            if len(outputs) > 0:
                min_dist = 400
                master_idx = -1
                for j, (output, conf) in enumerate(zip(outputs, confs)): 
                    bboxes = output[0:4]

                    dist = calc_distance(calc_center(bboxes), [center_w, center_h])
                    if dist < min_dist:
                        min_dist = dist
                        master_idx = j

                if master_idx != -1:
                    output = outputs[master_idx]
                    bboxes = output[0:4]
                    id = output[4]
                    cls = output[5]
                    label = "Master "+str(int(id))+" conf: {:.2f}".format(conf)

                    pts = np.array([[int(bboxes[2]), int(bboxes[3])], [int(bboxes[0]), int(bboxes[3])], [opt.cart_size[0], center_h*2], [opt.cart_size[1], center_h*2]], np.int32)
                    
                    if self.opt.track_visualization:
                        plot_one_box(bboxes, img_det, label=label, color=self.colors[int(cls)], line_thickness=2)
                        img_det = cv2.polylines(img_det, [pts], True, (211,0,148), 2)

                    blank_img = cv2.fillPoly(blank_img, [pts], (255))
                    blank_img_ori = blank_img.copy()
                    blank_img = cv2.bitwise_and(color_seg, blank_img)

                    for xyxy in obs_list:
                        blank_img = cv2.rectangle(blank_img, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0), -1)

                    ori_stat = np.sum(blank_img_ori == 255)
                    new_stat = np.sum(blank_img == 255)
                    road_check = new_stat/ori_stat

                    if road_check > opt.thres:
                        result = "DRIVE"

        else:
            self.strongsort.increment_ages()
            
        return img_det, result, road_check
    

    def inference(self, dataset):
        for i, (path, img, img_det, vid_cap, shapes) in tqdm(enumerate(dataset),total = len(dataset)):
#             t2 = time.time()
            img = transform(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
                
            # Inference
#             t1 = time.time()
            det_out, da_seg_out, _= self.model(img)
            inf_out, _ = det_out
#             print(time.time() - t1)

            # Apply NMS
            det_pred = non_max_suppression(inf_out, conf_thres=opt.conf_thres, iou_thres=opt.iou_thres, classes=None, agnostic=False)
            det = det_pred[0]

            save_path = str(opt.save_dir +'/'+ Path(path).name) if dataset.mode != 'stream' else str(opt.save_dir + '/' + "web.mp4")

            _, _, height, width = img.shape
            h,w,_=img_det.shape
            ori_img = img_det.copy()

            img_det, blank_img, color_seg, center_w, center_h = self.process_segmentation(img_det, da_seg_out, shapes, width, height)

            img_det, result, road_check = self.process_detection(det, img, ori_img, img_det, color_seg, blank_img, center_w, center_h)
                    
            if self.opt.visualization:
                img_det = cv2.resize(img_det, (w, h))
                
                if result == "DRIVE":
                    cv2.putText(img_det, "DRIVE", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 4, (255,0,0), 7)
                    spec = "da_ratio: {:.03f}".format(road_check)
                    cv2.putText(img_det, spec, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 3)

                else:
                    cv2.putText(img_det, "STOP", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 4, (0,0,255), 7)
#                     spec = "da_ratio: {:.03f}".format(road_check)
#                     cv2.putText(img_det, spec, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
#                 cv2.imwrite(os.path.join("/home/ubuntu/workspace/ywshin/construct/YOLOP/inference/demo_track/", "{:03d}.png".format(i)), img_det)

                if dataset.mode == 'images':
                    cv2.imwrite(save_path,img_det)

                elif dataset.mode == 'video':
                    if self.vid_path != save_path:  # new video
                        self.vid_path = save_path
                        if isinstance(self.vid_writer, cv2.VideoWriter):
                            self.vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        h,w,_=img_det.shape
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(img_det)

#             print(time.time() - t2)


#             print('Results saved to %s' % Path(opt.save_dir))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='/home/ubuntu/workspace/ywshin/construct/YOLOP/runs/Patrasche/221014_from_scratch/epoch-400.pth', help='model.pth path(s)')
    parser.add_argument('--source', type=str, default='inference/videos', help='source')  # file/folder   ex:inference/images
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='2', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-dir', type=str, default='inference/output', help='directory to save results')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--cart-size', type=int , default = (400, 1520), help="cart size")
    parser.add_argument('--thres', type=float , default = 0.9, help="road pass threshold")
    parser.add_argument('--strong-sort-weights', type=str, default="/home/ubuntu/workspace/ywshin/construct/YOLOP/strong_sort/weights/osnet_x0_25_msmt17.pt")
    
    parser.add_argument('--visualization', action='store_true', help="visualization")
    parser.add_argument('--seg-visualization', action='store_true', help="segmentation visualization")
    parser.add_argument('--track-visualization', action='store_true', help="detection and tracking visualization")

    opt = parser.parse_args()
    
    if opt.visualization:
        opt.seg_visualization = True
        opt.track_visualization = True
    
    if opt.seg_visualization or opt.track_visualization:
        opt.visualization = True
    
    
    with torch.no_grad():
        patrasche = Patrasche(cfg, opt)
        dataset = patrasche.set_dataloader()
        result = patrasche.inference(dataset)