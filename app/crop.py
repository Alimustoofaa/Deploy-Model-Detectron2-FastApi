import cv2
import uuid

def crop_img(outputs, im):
    boxes = {}
    score_array = []
    img_crop_array = []

    for i, coordinates in enumerate(outputs["instances"].to("cpu").pred_boxes, start=0):
        score_array.append(outputs["instances"].scores[i].item())
        coordinates_array = []
        for k in coordinates:
            coordinates_array.append(int(k))
        boxes[uuid.uuid4().hex[:].upper()] = coordinates_array

    for k,v in boxes.items():
        crop_img = im[v[1]-13:v[3]+13, v[0]-13:v[2]+13, :]
        img_crop_array.append(crop_img)

    score_hight = max(range(len(score_array)), key=score_array.__getitem__)
    img_detection = img_crop_array[score_hight]
    
    return img_detection