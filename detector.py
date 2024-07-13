import torch
import cv2
import numpy as np
from sympy.physics.vector import frame
from torch.autograd import Variable
from darknet import Darknet
from util import process_result, load_images, resize_image, cv_image2tensor, transform_result
import pickle as pkl
import argparse
import math
import random
import os.path as osp
import os
import sys
from datetime import datetime


def load_classes(namesfile):
    fp = open(namesfile, "r")
    names = fp.read().split("\n")[:-1]
    return names

def parse_args():
    parser = argparse.ArgumentParser(description='YOLOv3 object detection')
    parser.add_argument('-i', '--input', default=None, help='input image or directory or video')
    parser.add_argument('-t', '--obj-thresh', type=float, default=0.5, help='objectness threshold, DEFAULT: 0.5')
    parser.add_argument('-n', '--nms-thresh', type=float, default=0.4, help='non max suppression threshold, DEFAULT: 0.4')
    parser.add_argument('-o', '--outdir', default='detection', help='output directory, DEFAULT: detection/')
    parser.add_argument('-v', '--video', action='store_true', default=False, help='flag for detecting a video input')
    parser.add_argument('-w', '--webcam', action='store_true',  default=False, help='flag for detecting from webcam.')
    parser.add_argument('--cuda', action='store_true', default=False, help='flag for running on GPU')
    parser.add_argument('--no-show', action='store_true', default=False, help='do not show the detected video in real time')

    args = parser.parse_args()

    if args.webcam and args.input is not None:
        print("ERROR: Webcam mode doesn't require an input argument.")
        parser.print_help()
        sys.exit(1)

    return args



def load_palette():
    try:
        palette_path = "palette.pkl"
        if os.path.exists(palette_path):
            with open(palette_path, "rb") as f:
                colors = pkl.load(f)
            return colors
        else:
            print("Palette file not found:", palette_path)
            return None
    except Exception as e:
        print("Error loading palette:", e)
        return None


def create_batches(frames, batch_size):
    num_batches = math.ceil(len(frames) / batch_size)
    batches = [frames[i * batch_size: (i + 1) * batch_size] for i in range(num_batches)]
    return batches


def transform_result(detections, images, input_size):
    """
    Transform detection coordinates to original image size.
    """
    transformed_detections = []
    for detection in detections:
        if detection.ndimension() == 1:
            detection = detection.unsqueeze(0)

        img = images[0]  # Assuming batch size of 1 for simplicity
        orig_h, orig_w = img.shape[0], img.shape[1]
        inp_h, inp_w = input_size[0], input_size[1]

        scale = min(inp_w / orig_w, inp_h / orig_h)
        pad_x = (inp_w - orig_w * scale) / 2
        pad_y = (inp_h - orig_h * scale) / 2

        detection = detection.clone()  # Clone the tensor to perform out-of-place operations
        detection[:, [1, 3]] = (detection[:, [1, 3]] - pad_x) / scale
        detection[:, [2, 4]] = (detection[:, [2, 4]] - pad_y) / scale

        detection[:, [1, 3]] = detection[:, [1, 3]].clamp(0, orig_w)
        detection[:, [2, 4]] = detection[:, [2, 4]].clamp(0, orig_h)

        transformed_detections.append(detection)
    if len(transformed_detections) > 0:
        return torch.cat(transformed_detections, 0)
    else:
        return torch.tensor([])


def draw_bbox(imgs, bbox, colors, classes, original_size=None):
    img = imgs[0]

    if isinstance(bbox, torch.Tensor):
        bbox = bbox.detach().cpu().numpy()

    p1 = tuple(bbox[1:3].astype(int).tolist())
    p2 = tuple(bbox[3:5].astype(int).tolist())

    label = classes[int(bbox[-1])]

    if len(colors) == 0:
        print("Error: Colors list is empty.")
        return

    color = random.choice(colors)

    if not isinstance(color, tuple) or len(color) != 3:
        print("Error: Invalid color format.")
        return

    if img is None:
        print("Error: Invalid image.")
        return
    if label is None:
        print("Error: Invalid label.")
        return

    cv2.rectangle(img, p1, p2, color, 2)

    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)[0]

    if original_size is not None:
        text_position = (p1[0], p1[1] - text_size[1] - 4)
    else:
        text_position = (p1[0], p1[1] + 20)

    cv2.rectangle(img, text_position, (text_position[0] + text_size[0] + 4, text_position[1] + text_size[1] + 4), color, -1)
    cv2.putText(img, label, (p1[0], p1[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)










def detect_video(model, args):
    input_size = [int(model.net_info['height']), int(model.net_info['width'])]

    colors = pkl.load(open("palette.pkl", "rb"))
    classes = load_classes("data/coco.names")
    colors = [colors[1]]

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {args.input}.")
        return

    output_path = osp.join(args.outdir, 'det_' + osp.basename(args.input).rsplit('.')[0] + '.avi')

    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    read_frames = 0
    start_time = datetime.now()
    print('Detecting...')

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        orig_h, orig_w = frame.shape[:2]
        img_resized = cv2.resize(frame, (input_size[1], input_size[0]))
        frame_tensor = cv_image2tensor(img_resized, input_size).unsqueeze(0)
        frame_tensor = Variable(frame_tensor)

        if args.cuda:
            frame_tensor = frame_tensor.cuda()

        detections = model(frame_tensor, args.cuda).cpu()
        detections = process_result(detections, args.obj_thresh, args.nms_thresh)

        if len(detections) != 0:
            detections = transform_result(detections, [frame], input_size)

        for detection in detections:
            draw_bbox([frame], detection, colors, classes)

        out.write(frame)
        read_frames += 1

        if read_frames % 2 == 0:
            print('Number of frames processed:', read_frames)

    end_time = datetime.now()
    print('Detection finished in %s' % (end_time - start_time))
    print('Total frames:', read_frames)
    cap.release()
    out.release()
    cv2.destroyAllWindows()









def detect_image(model, args):
    print('Loading input image(s)...')
    colors = pkl.load(open("palette.pkl", "rb"))
    classes = load_classes("data/coco.names")
    input_size = [int(model.net_info['height']), int(model.net_info['width'])]
    batch_size = int(model.net_info['batch'])
    imlist, imgs = load_images(args.input)
    print('Input image(s) loaded')

    print("Length of imlist:", len(imlist))
    print("Length of imgs:", len(imgs))

    if not osp.exists(args.outdir):
        os.makedirs(args.outdir)

    start_time = datetime.now()
    print('Detecting...')
    for batchi, img_batch in enumerate(imgs):
        img_resized = cv2.resize(img_batch, (input_size[1], input_size[0]))  # Resize the image
        img_tensors = cv_image2tensor(img_resized, input_size)
        img_tensors = torch.unsqueeze(img_tensors, 0)  # Add batch dimension
        img_tensors = Variable(img_tensors)

        if args.cuda:
            img_tensors = img_tensors.cuda()

        detections = model(img_tensors, args.cuda).cpu()
        detections = process_result(detections, args.obj_thresh, args.nms_thresh)

        if len(detections) == 0:
            continue

        detections = transform_result(detections, [img_resized], input_size)
        original_size = (img_batch.shape[1], img_batch.shape[0])  # Get original image size
        for i, detection in enumerate(detections):
            save_path = osp.join(args.outdir, 'det_' + osp.basename(imlist[batchi * batch_size]))
            # Ensure detection is in the correct format before saving
            if isinstance(detection, torch.Tensor):
                detection = detection.detach().numpy()
            if isinstance(detection, np.ndarray):
                draw_bbox([img_resized], detection, colors, classes, original_size)  # Pass original size
                cv2.imwrite(save_path, img_resized)
            else:
                print("Detection result is not in the correct format.")
                continue

    end_time = datetime.now()
    print('Detection finished in %s' % (end_time - start_time))


def detect_webcam(model, args):
    cap = cv2.VideoCapture(0)  # Open the webcam
    colors = pkl.load(open("palette.pkl", "rb"))
    classes = load_classes("data/coco.names")
    colors = [colors[1]]

    while True:
        ret, frame = cap.read()  # Capture frame-by-frame

        # Perform object detection on the frame
        frame_tensor = cv_image2tensor(frame, [int(model.net_info['height']), int(model.net_info['width'])]).unsqueeze(
            0)
        frame_tensor = Variable(frame_tensor)
        if args.cuda:
            frame_tensor = frame_tensor.cuda()

        detections = model(frame_tensor, args.cuda).cpu()
        detections = process_result(detections, args.obj_thresh, args.nms_thresh)
        if len(detections) != 0:
            detections = transform_result(detections, [frame],
                                          [int(model.net_info['height']), int(model.net_info['width'])])
            for detection in detections:
                draw_bbox([frame], detection, colors, classes)

        # Display the resulting frame
        cv2.imshow('Webcam', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Quit on 'q' key press
            break

    # Release the webcam
    cap.release()
    cv2.destroyAllWindows()


def main():
    args = parse_args()

    if args.cuda and not torch.cuda.is_available():
        print("ERROR: cuda is not available, try running on CPU")
        sys.exit(1)

    print('Loading network...')
    model = Darknet("cfg/yolov3.cfg")
    model.load_weights('yolov3.weights')
    if args.cuda:
        model.cuda()

    model.eval()
    print('Network loaded')

    if args.webcam:
        detect_webcam(model, args)
    elif args.video:
        detect_video(model, args)
    elif args.input:
        detect_image(model, args)
    else:
        print("ERROR: Please specify either --webcam, --video, or --input for detection.")






if __name__ == '__main__':
    main()
