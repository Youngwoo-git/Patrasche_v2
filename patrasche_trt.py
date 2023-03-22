import argparse
import os, sys
import shutil
import time
from pathlib import Path
import imageio

sys.path.append("../patrasche/tensorrt/")
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
import tensorrt as trt

from lib.config import cfg
from lib.config import update_config
from lib.utils.utils import create_logger, select_device, time_synchronized
from lib.models import get_net
from lib.dataset import LoadImages, LoadStreams
from lib.core.general import non_max_suppression, scale_coords
from lib.utils import plot_one_box,show_seg_result
from lib.dataset.convert import obj_list
import samples.python.common as common
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
    def __init__(self, cfg, opt, TRT_LOGGER):
        self.cfg = cfg
        self.opt = opt
        
        device = select_device(None,opt.device)
        if os.path.exists(opt.save_dir):  # output dir
            shutil.rmtree(opt.save_dir)  # delete dir
        os.makedirs(opt.save_dir)
        half = False
        # half = device.type != 'cpu'

#         self.model = get_net(cfg)
#         checkpoint = torch.load(opt.weights, map_location= device)
#         self.model.load_state_dict(checkpoint['state_dict'])
#         self.model = self.model.to(device)
        
#         if half:
#             self.model.half()    
        trt.init_libnvinfer_plugins(TRT_LOGGER, "")
        with open(opt.weights, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())

        self.context = engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = common.allocate_buffers(engine)
            
        
        self.names = obj_list
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.names))]
        
        self.vid_path, self.vid_writer = None, None
        
        img = np.array((1, 3, 384, 640)).astype(np.float32)  # init img
#         img = np.array((1, 3, 192, 320)).astype(np.float32)  # init img
        self.inputs[0].host = img
        _ = common.do_inference_v2(self.context, bindings=self.bindings, inputs=self.inputs, outputs=self.outputs, stream=self.stream)
        # img = torch.zeros((1, 3, opt.img_size, opt.img_size), device=device)  # init img
        # _ = self.model(img.half() if half else img) if device.type != 'cpu' else None  # run once
        # self.model.eval()
        
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
#                 elif visualization: #만약 영상을 사람 1명만 있는 것으로 실행시킬 계획이라면
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

            outputs = self.strongsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), ori_img)
            
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
    
    
    def trt_to_torch(self, trt_outputs):
#         for 640*384 weight
        det_0 = torch.from_numpy(np.reshape(trt_outputs[3], (1,15120,101)))
        det_1_0 = torch.from_numpy(np.reshape(trt_outputs[0], (1,3,48,80,101)))
        det_1_1 = torch.from_numpy(np.reshape(trt_outputs[1], (1,3,24,40,101)))
        det_1_2 = torch.from_numpy(np.reshape(trt_outputs[2], (1,3,12,20,101)))
        da_seg = torch.from_numpy(np.reshape(trt_outputs[4], (1,2,384,640)))
        ll_seg = torch.from_numpy(np.reshape(trt_outputs[5], (1,2,384,640)))
        
#         for 320*192 weight
#         det_0 = torch.from_numpy(np.reshape(trt_outputs[3], (1,3780,101)))
#         det_1_0 = torch.from_numpy(np.reshape(trt_outputs[0], (1,3,24,40,101)))
#         det_1_1 = torch.from_numpy(np.reshape(trt_outputs[1], (1,3,12,20,101)))
#         det_1_2 = torch.from_numpy(np.reshape(trt_outputs[2], (1,3,6,10,101)))
#         da_seg = torch.from_numpy(np.reshape(trt_outputs[4], (1,2,192,320)))
#         ll_seg = torch.from_numpy(np.reshape(trt_outputs[5], (1,2,192,320)))
        
        return (det_0, [det_1_0, det_1_1, det_1_2]), da_seg, ll_seg

    def inference(self, dataset):
        for i, (path, img, img_det, vid_cap, shapes) in tqdm(enumerate(dataset),total = len(dataset)):
            t2 = time.time()
            img = transform(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            # Inference
            img = img.detach().cpu().numpy().astype(np.float32)
            
            self.inputs[0].host = img
            # t1 = time.time()
            trt_outputs = common.do_inference_v2(self.context, bindings=self.bindings, inputs=self.inputs, outputs=self.outputs, stream=self.stream)
            # print(time.time() - t1)

            det_out, da_seg_out, _= self.trt_to_torch(trt_outputs)

            # det_out, da_seg_out, _= self.model(img)
            inf_out, _ = det_out
#             
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
                    cv2.putText(img_det, "DRIVE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
                    spec = "da_ratio: {:.03f}".format(road_check)
                    cv2.putText(img_det, spec, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

                else:
                    cv2.putText(img_det, "STOP", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
#                     spec = "da_ratio: {:.03f}".format(road_check)
#                     cv2.putText(img_det, spec, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

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

            print(time.time() - t2)


#             print('Results saved to %s' % Path(opt.save_dir))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='weights/yolop.trt', help='model.pth path(s)')
    parser.add_argument('--source', type=str, default='inference/videos', help='source')  # file/folder   ex:inference/images
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
#     parser.add_argument('--img-size', type=int, default=320, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-dir', type=str, default='inference/output', help='directory to save results')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--cart-size', type=int , default = (400, 1520), help="cart size")
    parser.add_argument('--thres', type=float , default = 0.9, help="road pass threshold")
#     parser.add_argument('--strong-sort-weights', type=str, default="strong_sort/weights/osnet_x0_25_msmt17.pt")
    parser.add_argument('--strong-sort-weights', type=str, default="weights/torchreid_256X128_bs_16.trt")
    
    parser.add_argument('--visualization', action='store_true', help="visualization")
    parser.add_argument('--seg-visualization', action='store_true', help="segmentation visualization")
    parser.add_argument('--track-visualization', action='store_true', help="detection and tracking visualization")

    opt = parser.parse_args()
    
    if opt.visualization:
        opt.seg_visualization = True
        opt.track_visualization = True
    
    if opt.seg_visualization or opt.track_visualization:
        opt.visualization = True
    
    TRT_LOGGER = trt.Logger()
    with torch.no_grad():
        patrasche = Patrasche(cfg, opt, TRT_LOGGER)
        dataset = patrasche.set_dataloader()
        result = patrasche.inference(dataset)