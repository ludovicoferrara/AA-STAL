import time
import numpy as np
import os
from collections import defaultdict
import torch
import json

from dataset.ava import AVA_Dataset

from .ava_eval_helper import (
    run_evaluation,
    read_csv,
    read_exclusions,
    read_labelmap,
    write_results
)



class AVA_Evaluator(object):
    def __init__(self, 
                 d_cfg,
                 data_root,
                 img_size,
                 len_clip,
                 sampling_rate,
                 batch_size,
                 transform,
                 collate_fn,
                 full_test_on_val=False,
                 version='v2.2'):
        self.all_preds = []
        self.full_ava_test = full_test_on_val
        self.data_root = data_root
        self.backup_dir = d_cfg['backup_dir']
        self.annotation_dir = os.path.join(data_root, d_cfg['annotation_dir'])
        self.labelmap_file = os.path.join(self.annotation_dir, d_cfg['labelmap_file'])
        self.frames_dir = os.path.join(data_root, d_cfg['frames_dir'])
        self.frame_list = os.path.join(data_root, d_cfg['frame_list'])
        self.exclusion_file = os.path.join(self.annotation_dir, d_cfg['val_exclusion_file'])
        self.gt_box_list = os.path.join(self.annotation_dir, d_cfg['val_gt_box_list'])

        # load data
        self.excluded_keys = read_exclusions(self.exclusion_file)
        self.categories, self.class_whitelist = read_labelmap(self.labelmap_file)
        self.full_groundtruth = read_csv(self.gt_box_list, self.class_whitelist)
        self.mini_groundtruth = self.get_ava_mini_groundtruth(self.full_groundtruth)
        _, self.video_idx_to_name = self.load_image_lists(self.frames_dir, self.frame_list, is_train=False)

        # create output_json file
        os.makedirs(self.backup_dir, exist_ok=True)
        self.backup_dir = os.path.join(self.backup_dir, 'ava_{}'.format(version))
        os.makedirs(self.backup_dir, exist_ok=True)
        self.output_json = os.path.join(self.backup_dir, 'ava_detections.json')

        # dataset
        self.testset = AVA_Dataset(
            cfg=d_cfg,
            data_root=data_root,
            is_train=False,
            img_size=img_size,
            transform=transform,
            len_clip=len_clip,
            sampling_rate=sampling_rate
        )
        self.num_classes = self.testset.num_classes

        # dataloader
        self.testloader = torch.utils.data.DataLoader(
            dataset=self.testset, 
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn, 
            num_workers=2, #CHANGED FOR COLAB
            drop_last=False,
            pin_memory=True
            )
    

    def get_ava_mini_groundtruth(self, full_groundtruth):
        """
        Get the groundtruth annotations corresponding the "subset" of AVA val set.
        We define the subset to be the frames such that (second % 4 == 0).
        We optionally use subset for faster evaluation during training
        (in order to track training progress).
        Args:
            full_groundtruth(dict): list of groundtruth.
        """
        return full_groundtruth
        #ret = [defaultdict(list), defaultdict(list), defaultdict(list)]
		
        #for i in range(3):
        #    for key in full_groundtruth[i].keys():
        #        if int(key.split(",")[1]) % 4 == 0:
        #            ret[i][key] = full_groundtruth[i][key]
        #return ret

    def filter_background_detections(self, detections, groundtruth, iou_thresh=0.5):
        """
        Filtra le predizioni rimuovendo i bounding box che non matchano
        spazialmente con le annotazioni GT (ignorando il background).
        """
        det_boxes, det_labels, det_scores = detections
        gt_boxes = groundtruth[0]

        filtered_boxes = defaultdict(list)
        filtered_labels = defaultdict(list)
        filtered_scores = defaultdict(list)

        for key in det_boxes.keys():
            if key not in gt_boxes:
                continue  # Salta se non c'è GT per questo frame

            for i in range(len(det_boxes[key])):
                p_box = det_boxes[key][i]
                max_iou = 0
                
                for g_box in gt_boxes[key]:
                    # Calcolo IoU per coordinate [y1, x1, y2, x2]
                    y1 = max(p_box[0], g_box[0])
                    x1 = max(p_box[1], g_box[1])
                    y2 = min(p_box[2], g_box[2])
                    x2 = min(p_box[3], g_box[3])
                    
                    inter = max(0, x2 - x1) * max(0, y2 - y1)
                    area_p = max(0, p_box[2] - p_box[0]) * max(0, p_box[3] - p_box[1])
                    area_g = max(0, g_box[2] - g_box[0]) * max(0, g_box[3] - g_box[1])
                    
                    iou = inter / (area_p + area_g - inter + 1e-6)
                    if iou > max_iou:
                        max_iou = iou
                
                # Mantiene la predizione solo se matcha con la persona in GT
                if max_iou >= iou_thresh:
                    filtered_boxes[key].append(p_box)
                    filtered_labels[key].append(det_labels[key][i])
                    filtered_scores[key].append(det_scores[key][i])

        return filtered_boxes, filtered_labels, filtered_scores
      
      
    def load_image_lists(self, frames_dir, frame_list, is_train):
        """
        Loading image paths from corresponding files.

        Args:
            frames_dir (str): path to frames dir.
            frame_list (str): path to frame list.
            is_train (bool): if it is training dataset or not.

        Returns:
            image_paths (list[list]): a list of items. Each item (also a list)
                corresponds to one video and contains the paths of images for
                this video.
            video_idx_to_name (list): a list which stores video names.
        """
        # frame_list_dir is /data3/ava/frame_lists/
        # contains 'train.csv' and 'val.csv'
        if is_train:
            list_name = "train.csv"
        else:
            list_name = "val.csv"

        list_filename = os.path.join(frame_list, list_name)

        image_paths = defaultdict(list)
        video_name_to_idx = {}
        video_idx_to_name = []
        with open(list_filename, "r") as f:
            f.readline()
            for line in f:
                row = line.split()
                # The format of each row should follow:
                # original_vido_id video_id frame_id path labels.
                assert len(row) == 5
                video_name = row[0]

                if video_name not in video_name_to_idx:
                    idx = len(video_name_to_idx)
                    video_name_to_idx[video_name] = idx
                    video_idx_to_name.append(video_name)

                data_key = video_name_to_idx[video_name]

                image_paths[data_key].append(os.path.join(frames_dir, row[3]))

        image_paths = [image_paths[i] for i in range(len(image_paths))]

        print("Finished loading image paths from: {}".format(list_filename))

        return image_paths, video_idx_to_name


    def update_stats(self, preds):
        self.all_preds.extend(preds)


    def get_ava_eval_data(self):
        out_scores = defaultdict(list)
        out_labels = defaultdict(list)
        out_boxes = defaultdict(list)
        count = 0

        # each pred is [[x1, y1, x2, y2], cls_out, [video_idx, src]]
        for i in range(len(self.all_preds)):
            pred = self.all_preds[i]
            assert len(pred) == 3
            video_idx = int(np.round(pred[-1][0]))
            sec = int(np.round(pred[-1][1]))
            box = pred[0]
            scores = pred[1]
            assert len(scores) == 15

            video = self.video_idx_to_name[video_idx]
            key = video + ',' + "%04d" % (sec)
            box = [box[1], box[0], box[3], box[2]]  # turn to y1,x1,y2,x2

            for cls_idx, score in enumerate(scores):
                if cls_idx + 1 in self.class_whitelist:
                    out_scores[key].append(score)
                    out_labels[key].append(cls_idx + 1)
                    out_boxes[key].append(box)
                    count += 1

        return out_boxes, out_labels, out_scores


    def calculate_mAP(self, epoch):
        eval_start = time.time()
        detections = self.get_ava_eval_data()
        if self.full_ava_test:
            groundtruth = self.full_groundtruth
        else:
            groundtruth = self.mini_groundtruth
            
        detections = self.filter_background_detections(detections, groundtruth, iou_thresh=0.5)

        print("Evaluating with %d unique GT frames." % len(groundtruth[0]))
        print("Evaluating with %d unique detection frames" % len(detections[0]))

        write_results(detections, os.path.join(self.backup_dir, "detections_{}.csv".format(epoch)))
        write_results(groundtruth, os.path.join(self.backup_dir, "groundtruth_{}.csv".format(epoch)))
        results = run_evaluation(self.categories, groundtruth, detections, self.excluded_keys)
        with open(self.output_json, 'w') as fp:
            json.dump(results, fp)
        print("Save eval results in {}".format(self.output_json))

        print("AVA eval done in %f seconds." % (time.time() - eval_start))

        return results["PascalBoxes_Precision/mAP@0.5IOU"]


    def evaluate_frame_map(self, model, epoch=1):
        model.eval()
        epoch_size = len(self.testloader)

        for iter_i, (_, batch_video_clip, batch_target) in enumerate(self.testloader):

            # to device
            batch_video_clip = batch_video_clip.to(model.device)

            with torch.no_grad():
                # inference
                batch_bboxes = model(batch_video_clip)

                # process batch
                preds_list = []
                for bi in range(len(batch_bboxes)):
                    bboxes = batch_bboxes[bi]
                    target = batch_target[bi]

                    # video info
                    video_idx = target['video_idx']
                    sec = target['sec']

                    for bbox in bboxes:
                        x1, y1, x2, y2 = bbox[:4]
                        det_conf = float(bbox[4])
                        cls_out = [np.sqrt(det_conf * cls_conf) for cls_conf in bbox[5:]]

                        preds_list.append([[x1,y1,x2,y2], cls_out, [video_idx, sec]])

            self.update_stats(preds_list)
            if iter_i % 100 == 0:
                log_info = "[%d / %d]" % (iter_i, epoch_size)
                print(log_info, flush=True)

        mAP = self.calculate_mAP(epoch)
        print("mAP: {}".format(mAP))

        # clear
        del self.all_preds
        self.all_preds = []

        return mAP
