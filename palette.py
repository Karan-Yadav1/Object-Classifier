import pickle as pkl

def create_palette():
    # Define class labels
    class_labels = [
        "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
        "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
        "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
        "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
        "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
        "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
        "toothbrush"
    ]

    # Generate RGB colors for each class label
    colors = [(int(x * 255 / len(class_labels)), int(y * 255 / len(class_labels)), int(z * 255 / len(class_labels)))
              for x, y, z in zip(range(len(class_labels)), range(len(class_labels)), range(len(class_labels)))]

    # Save the colors to the palette file
    with open('palette.pkl', 'wb') as f:
        pkl.dump(colors, f)

# Create the palette file
create_palette()
