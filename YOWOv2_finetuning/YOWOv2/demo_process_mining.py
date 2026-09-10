import argparse
import cv2
import os
import time
import numpy as np
import torch
import imageio
import csv
from PIL import Image

from dataset.transforms import BaseTransform
from utils.misc import load_weight
from utils.box_ops import rescale_bboxes
from utils.vis_tools import vis_detection
from config import build_dataset_config, build_model_config
from models import build_model


def parse_args():
    parser = argparse.ArgumentParser(description='YOWOv2 Demo')

    # basic
    parser.add_argument('-size', '--img_size', default=224, type=int,
                        help='the size of input frame')
    parser.add_argument('--show', action='store_true', default=False,
                        help='show the visulization results.')
    parser.add_argument('--cuda', action='store_true', default=False, 
                        help='use cuda.')
    parser.add_argument('--save_folder', default='det_results/', type=str,
                        help='Dir to save results')
    parser.add_argument('-vs', '--vis_thresh', default=0.3, type=float,
                        help='threshold for visualization')
    parser.add_argument('--video', default='video.mp4', type=str,
                        help='AVA video name.')
    parser.add_argument('--gif', action='store_true', default=False, 
                        help='generate gif.')
    # Nuovo parametro per il logging
    parser.add_argument('--log_freq', default=16, type=int,
                        help='Frequenza di inferenza e logging in frame (default: 16)')

    # class label config
    parser.add_argument('-d', '--dataset', default='ava_v2.2',
                        help='ava_v2.2')
    parser.add_argument('--pose', action='store_true', default=False, 
                        help='show 14 action pose of AVA.')

    # model
    parser.add_argument('-v', '--version', default='yowo_v2_large', type=str,
                        help='build YOWOv2')
    parser.add_argument('--weight', default=None,
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('-ct', '--conf_thresh', default=0.1, type=float,
                        help='confidence threshold')
    parser.add_argument('-nt', '--nms_thresh', default=0.5, type=float,
                        help='NMS threshold')
    parser.add_argument('--topk', default=40, type=int,
                        help='NMS threshold')
    parser.add_argument('-K', '--len_clip', default=16, type=int,
                        help='video clip length.')
    parser.add_argument('-m', '--memory', action="store_true", default=False,
                        help="memory propagate.")

    return parser.parse_args()
                    

def multi_hot_vis(args, frame, out_bboxes, orig_w, orig_h, class_names, act_pose=False):
    frame_logs = []
    if out_bboxes is None:
        return frame, frame_logs

    person_counter = 1
    # visualize detection results
    for bbox in out_bboxes:
        x1, y1, x2, y2 = bbox[:4]
        if act_pose:
            # only show 14 poses of AVA.
            cls_conf = bbox[5:5+14]
        else:
            # show all actions of AVA.
            cls_conf = bbox[5:]
    
        # rescale bbox
        x1, x2 = int(x1 * orig_w), int(x2 * orig_w)
        y1, y2 = int(y1 * orig_h), int(y2 * orig_h)

        # score = obj * cls
        det_conf = float(bbox[4])
        cls_scores = np.sqrt(det_conf * cls_conf)

        indices = np.where(cls_scores > args.vis_thresh)
        scores = cls_scores[indices]
        indices = list(indices[0])
        scores = list(scores)

        if len(scores) > 0:
            # LOGICA CSV: estrae stringa e ID
            action_names = [str(class_names[cls_ind]) for cls_ind in indices]
            actions_str = "-".join(action_names)
            person_id = f"Person_{person_counter}"
            frame_logs.append((actions_str, person_id))
            person_counter += 1

            # draw bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # draw text
            blk   = np.zeros(frame.shape, np.uint8)
            font  = cv2.FONT_HERSHEY_SIMPLEX
            coord = []
            text  = []
            text_size = []

            for _, cls_ind in enumerate(indices):
                text.append("[{:.2f}] ".format(scores[_]) + str(class_names[cls_ind]))
                text_size.append(cv2.getTextSize(text[-1], font, fontScale=0.5, thickness=1)[0])
                coord.append((x1+3, y1+14+20*_))
                cv2.rectangle(blk, (coord[-1][0]-1, coord[-1][1]-12), (coord[-1][0]+text_size[-1][0]+1, coord[-1][1]+text_size[-1][1]-4), (0, 255, 0), cv2.FILLED)
            frame = cv2.addWeighted(frame, 1.0, blk, 0.5, 1)
            for t in range(len(text)):
                cv2.putText(frame, text[t], coord[t], font, 0.5, (0, 0, 0), 1)
    
    return frame, frame_logs


@torch.no_grad()
def detect(args, model, device, transform, class_names, class_colors):
    # path to save 
    save_path = os.path.join(args.save_folder, 'demo', 'videos')
    os.makedirs(save_path, exist_ok=True)

    # path to video
    path_to_video = os.path.join(args.video)

    # video
    video = cv2.VideoCapture(path_to_video)
    
    # Calcolo FPS dal video per determinare il tempo esatto
    video_fps = video.get(cv2.CAP_PROP_FPS)
    if video_fps == 0 or np.isnan(video_fps):
        video_fps = 30.0  # Fallback di sicurezza
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    save_size = (960, 720)
    save_name = os.path.join(save_path, 'detection.avi')
    out = cv2.VideoWriter(save_name, fourcc, video_fps, save_size)

    # File CSV per log in real-time
    csv_path = os.path.join(save_path, 'action_log.csv')
    csv_file = open(csv_path, mode='w', newline='')
    csv_writer = csv.writer(csv_file, delimiter=',')
    csv_writer.writerow(['Time(HH:MM:SS)', 'Action', 'person'])

    # run
    video_clip = []
    image_list = []
    frame_count = 0
    last_bboxes = None

    while(True):
        ret, frame = video.read()
        
        if ret:
            frame_count += 1
            # to RGB
            frame_rgb = frame[..., (2, 1, 0)]
            # to PIL image
            frame_pil = Image.fromarray(frame_rgb.astype(np.uint8))

            # prepare
            if len(video_clip) <= 0:
                for _ in range(args.len_clip):
                    video_clip.append(frame_pil)

            video_clip.append(frame_pil)
            del video_clip[0]

            # orig size
            orig_h, orig_w = frame.shape[:2]

            # Esegue inferenza e scrive log SOLO ai frame corrispondenti a log_freq
            if frame_count % args.log_freq == 0:
                # transform
                x, _ = transform(video_clip)
                # List [T, 3, H, W] -> [3, T, H, W]
                x = torch.stack(x, dim=1)
                x = x.unsqueeze(0).to(device) # [B, 3, T, H, W], B=1

                t0 = time.time()
                # inference
                outputs = model(x)
                
                # Calcolo Timestamp HH:MM:SS basato sul framerate del video
                current_seconds = int(frame_count / video_fps)
                hours = current_seconds // 3600
                minutes = (current_seconds % 3600) // 60
                seconds = current_seconds % 60
                timestamp_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

                if args.dataset in ['ava_v2.2']:
                    last_bboxes = outputs[0]
                    # Multi hot e Log CSV
                    frame, logs = multi_hot_vis(args, frame, last_bboxes, orig_w, orig_h, class_names, args.pose)
                    for log in logs:
                        csv_writer.writerow([timestamp_str, log[0], log[1]])
                    csv_file.flush() # Forza la scrittura per letture real-time da applicativi esterni
                    
            else:
                # Disegna le ultime detection conosciute per mantenere il video output continuo (nessun log)
                if args.dataset in ['ava_v2.2'] and last_bboxes is not None:
                    frame, _ = multi_hot_vis(args, frame, last_bboxes, orig_w, orig_h, class_names, args.pose)

            # save
            frame_resized = cv2.resize(frame, save_size)
            out.write(frame_resized)

            if args.gif:
                gif_resized = cv2.resize(frame, (200, 150))
                gif_resized_rgb = gif_resized[..., (2, 1, 0)]
                image_list.append(gif_resized_rgb)

            if args.show:
                # show
                cv2.imshow('key-frame detection', frame)
                cv2.waitKey(1)

        else:
            break

    video.release()
    out.release()
    csv_file.close()
    #cv2.destroyAllWindows()

    # generate GIF
    if args.gif:
        save_name = os.path.join(save_path, 'detect.gif')
        print('generating GIF ...')
        imageio.mimsave(save_name, image_list, fps=video_fps)
        print('GIF done: {}'.format(save_name))


if __name__ == '__main__':
    np.random.seed(100)
    args = parse_args()

    # cuda
    if args.cuda:
        print('use cuda')
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # config
    d_cfg = build_dataset_config(args)
    m_cfg = build_model_config(args)

    class_names = d_cfg['label_map']
    num_classes = d_cfg['valid_num_classes']

    class_colors = [(np.random.randint(255),
                     np.random.randint(255),
                     np.random.randint(255)) for _ in range(num_classes)]

    # transform
    basetransform = BaseTransform(img_size=args.img_size)

    # build model
    model, _ = build_model(
        args=args,
        d_cfg=d_cfg,
        m_cfg=m_cfg,
        device=device, 
        num_classes=num_classes, 
        trainable=False
        )

    # load trained weight
    model = load_weight(model=model, path_to_ckpt=args.weight)

    # to eval
    model = model.to(device).eval()

    # run
    detect(args=args,
            model=model,
            device=device,
            transform=basetransform,
            class_names=class_names,
            class_colors=class_colors)