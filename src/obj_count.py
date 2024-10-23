from ultralytics import YOLO
import numpy as np
from PIL import Image
import numpy as np
from collections import Counter, defaultdict
import json
model = YOLO('D:/tfolder/codingFile/AIlearning/AIchallenge/YOLO/resource/yolov8n.pt')     #pre trained yolo module, it will be downloaded to pc
model_size = (800,600)      #required img size for model preprocessing


#Lưu ý: ảnh vào ở dạng PIL
#save và load đưa thêm folder vào

#hàm trả về dictionary gồm 'class': 'số lượng', 'vị trí center rect', 'dài rộng rect'
def img_object_data(image):
    """
    Detect objects in the image using the YOLO model and return their details.
    
    Args:
    - image: np.array, the input image in OpenCV format
    - model: the YOLO model
    
    Returns:
    - results_dict: dict, containing object details including count, centers, and sizes
    """
    image = image.resize(model_size, Image.BICUBIC)
    results = model(image, show = True)
    # Extract class names from the model
    class_names = model.names

    if len(results) > 0 and hasattr(results[0], 'boxes'):
        # Extract detections
        detections = results[0].boxes.data.numpy()  # Each row in `detections` contains [x1, y1, x2, y2, confidence, class_id]

        # Extract detected class indices and bounding box coordinates
        detected_classes = detections[:, 5].astype(int)
        bounding_boxes = detections[:, :4]  # [x1, y1, x2, y2]

        # Count occurrences of each object
        counts = Counter(detected_classes)

        # Prepare the results
        results_dict = {}
        for class_id in counts.keys():
            class_name = class_names[class_id]
            object_boxes = bounding_boxes[detected_classes == class_id]
            centers = []
            sizes = []
            for box in object_boxes:
                x1, y1, x2, y2 = box
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                width = x2 - x1
                height = y2 - y1
                centers.append([center_x, center_y])
                sizes.append([width, height])

            results_dict[class_name] = {
                "count": counts[class_id],
                "centers": np.array(centers),
                "sizes": np.array(sizes)
            }

        return results_dict
    else:
        # Handle the case where there are no detections
        return {}


#lưu dictionary vào file json
def save_dict_to_file(data_dict, file_path):
    # Convert NumPy arrays to lists for JSON serialization
    for key, value in data_dict.items():
        value["centers"] = value["centers"].tolist()
        value["sizes"] = value["sizes"].tolist()
    
    with open(file_path, 'w') as f:
        json.dump(data_dict, f)

#load từ file json, trả về dict
def load_dict_from_file(file_path):
    with open(file_path, 'r') as f:
        data_dict = json.load(f)
    
    # Convert lists back to NumPy arrays
    for key, value in data_dict.items():
        value["centers"] = np.array(value["centers"])
        value["sizes"] = np.array(value["sizes"])
    
    return data_dict


