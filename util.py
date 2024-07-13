import torch
import os.path as osp
import os
import sys
import cv2
import numpy as np

# conduct objectness score filtering and non max supperssion
def process_result(detection, obj_threshhold, nms_threshhold):
    detection = to_corner(detection)
    output = torch.tensor([], dtype=torch.float)

    # do it batchwise and classwise
    for batchi in range(detection.size(0)):
        bboxes = detection[batchi]
        bboxes = bboxes[bboxes[:, 4] > obj_threshhold]

        if len(bboxes) == 0:
            continue

        # attributes of each bounding box: x1, y1, x2, y2, objectness score, prediction score, prediction index
        pred_score, pred_index = torch.max(bboxes[:, 5:], 1)
        pred_score = pred_score.unsqueeze(-1)
        pred_index = pred_index.float().unsqueeze(-1)
        bboxes = torch.cat((bboxes[:, :5], pred_score, pred_index), dim=1)
        pred_classes = torch.unique(bboxes[:, -1])

        # non max suppression for each predicted class

        for cls in pred_classes:
            bboxes_cls = bboxes[bboxes[:, -1] == cls]   # select boxes that predict the class
            _, sort_indices = torch.sort(bboxes_cls[:, 4], descending=True)
            bboxes_cls = bboxes_cls[sort_indices]   # sort by objectness score

            # select the box with the highest score and get rid of intercepting boxes with big IOU
            boxi = 0
            while boxi + 1 < bboxes_cls.size(0):
                ious = compute_ious(bboxes_cls[boxi], bboxes_cls[boxi+1:])
                bboxes_cls = torch.cat([bboxes_cls[:boxi+1], bboxes_cls[boxi+1:][ious < nms_threshhold]])
                boxi += 1

            # add batch index as the first attribute
            batch_idx_add = torch.full((bboxes_cls.size(0), 1), batchi)
            bboxes_cls = torch.cat((batch_idx_add, bboxes_cls), dim=1)
            output = torch.cat((output, bboxes_cls))

    return output

def to_corner(bboxes):
    newbboxes = bboxes.clone()
    newbboxes[:, :, 0] = bboxes[:, :, 0] - bboxes[:, :, 2] / 2
    newbboxes[:, :, 1] = bboxes[:, :, 1] - bboxes[:, :, 3] / 2
    newbboxes[:, :, 2] = bboxes[:, :, 0] + bboxes[:, :, 2] / 2
    newbboxes[:, :, 3] = bboxes[:, :, 1] + bboxes[:, :, 3] / 2
    return newbboxes

def compute_ious(target_box, comp_boxes):
    targetx1, targety1, targetx2, targety2 = target_box[:4]
    compx1s, compy1s, compx2s, compy2s = comp_boxes[:, :4].transpose(0, 1)

    interceptx1s = torch.max(targetx1, compx1s)
    intercepty1s = torch.max(targety1, compy1s)
    interceptx2s = torch.min(targetx2, compx2s)
    intercepty2s = torch.min(targety2, compy2s)

    intercept_areas = torch.clamp(interceptx2s - interceptx1s + 1, 0) * torch.clamp(intercepty2s - intercepty1s + 1, 0)

    target_area = (targetx2 - targetx1 + 1) * (targety2 - targety1 + 1)
    comp_areas = (compx2s - compx1s + 1) * (compy2s - compy1s + 1)

    union_areas = comp_areas + target_area - intercept_areas

    ious = intercept_areas / union_areas
    return ious


def load_images(impath):
    imgs = []
    if osp.isdir(impath):
        imlist = [osp.join(impath, img) for img in os.listdir(impath)]
    elif osp.isfile(impath):
        # Check if the input file is a video file
        if impath.lower().endswith(('.mp4', '.avi', '.mov')):
            cap = cv2.VideoCapture(impath)
            imlist = [impath] * int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                imgs.append(frame)
            cap.release()
        else:
            imlist = [impath]
            img = cv2.imread(impath)
            imgs.append(img)
    else:
        print('%s is not a valid path' % impath)
        sys.exit(1)

    return imlist, imgs




def cv_image2tensor(img, size):
    print("Original image shape:", img.shape)
    # Convert grayscale image to RGB
    if len(img.shape) < 3 or img.shape[2] == 1:  # Grayscale image
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:  # RGBA image
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    img = resize_image(img, size)
    print("Resized image shape:", img.shape)
    img = img.transpose((2, 0, 1)).copy()
    img = torch.from_numpy(img).float() / 255.0
    print("Tensor shape:", img.shape)
    # Check if img is None
    if img is None:
        print("Error: Converted image is None")
        return None

    return img



# resize_image by scaling while preserving aspect ratio and then padding remaining area with gray pixels
def resize_image(img, size):
    new_img = cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)
    if new_img is None:
        print("Error: Resized image is None")
    return new_img




# transform bouning box position in the resized image(input image to the network) to the corresponding position in the original image
def transform_result(detections, imgs, input_size):
    # get the original image dimensions
    img_dims = [[img.shape[0], img.shape[1]] for img in imgs]
    img_dims = torch.tensor(img_dims, dtype=torch.float)
    img_dims = torch.index_select(img_dims, 0, detections[:, 0].long())

    input_size = torch.tensor(input_size, dtype=torch.float)

    scale_factors = torch.min(input_size / img_dims, 1)[0].unsqueeze(-1)
    detections[:, [1, 3]] -= (input_size[1] - scale_factors * img_dims[:, 1].unsqueeze(-1)) / 2
    detections[:, [2, 4]] -= (input_size[0] - scale_factors * img_dims[:, 0].unsqueeze(-1)) / 2

    detections[:, 1:5] /= scale_factors

    # clipping
    detections[:, 1:5] = torch.clamp(detections[:, 1:5], 0)
    detections[:, [1, 3]] = torch.min(detections[:, [1, 3]], img_dims[:, 1].unsqueeze(-1))
    detections[:, [2, 4]] = torch.min(detections[:, [2, 4]], img_dims[:, 0].unsqueeze(-1))

    return detections