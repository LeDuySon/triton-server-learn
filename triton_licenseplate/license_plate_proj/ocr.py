import cv2
import numpy as np
from typing import List
from skimage.filters import threshold_local

from boundingbox import BoundingBox
from labels import LicensePlateTypes
import torch
import cv2

from torchvision import transforms

def show_img(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def detect_license_plate_type(text_regions, ratio=0.8):
    min_y = 416
    max_y = 0 
    avg_char_height = 0
    
    for bbox in text_regions:
        x, y, w, h = bbox
        max_y = max(max_y, y)
        min_y = min(min_y, y)
        
        avg_char_height += h
        
    avg_char_height /= len(text_regions)
    
    lp_type = LicensePlateTypes.ONE_LINE # bien 1 dong``
    if((max_y - min_y) / h > ratio):
        lp_type = LicensePlateTypes.TWO_LINE # bien 2 dong
        
    return lp_type
    
def get_chars_region(crop_img):
    print("Crop image size: ", crop_img.size)
    v_channel = cv2.split(cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV))[2]
        
    binary_thresh = threshold_local(v_channel, 15, offset=10, method="gaussian")
    thresh = (v_channel > binary_thresh).astype("uint8") * 255
    
    thresh = cv2.bitwise_not(thresh)
    
    output = cv2.connectedComponentsWithStats(
        thresh, 8, cv2.CV_32S)
    (num_labels, labels, stats, centroids) = output
    
    text_regions = []
    for idx in range(num_labels):
        if idx == 0:
            continue
        
        # get bbox
        x = stats[idx, cv2.CC_STAT_LEFT]
        y = stats[idx, cv2.CC_STAT_TOP]
        w = stats[idx, cv2.CC_STAT_WIDTH]
        h = stats[idx, cv2.CC_STAT_HEIGHT]
        area = stats[idx, cv2.CC_STAT_AREA]
        
        # filter candidates
        box_ratio = w / float(h)
        heigh_ratio = h / float(crop_img.shape[0])
        
        if(0.2 < box_ratio < 1 and 0.3 < heigh_ratio < 2):
            text_regions.append((x, y, w, h))
    
    lp_type = detect_license_plate_type(text_regions)
    
    if(lp_type == LicensePlateTypes.TWO_LINE):
        text_regions = sorted(text_regions, key=lambda x: x[1])
        text_regions = sorted(text_regions[:4], key=lambda x: x[0]) + sorted(text_regions[4:], key=lambda x: x[0])
        
    elif(lp_type == LicensePlateTypes.ONE_LINE):
        text_regions = sorted(text_regions, lambda x: x[0])
    
    return text_regions
    
def get_text_from_img(img: np.ndarray, bboxes: List[BoundingBox]):
    print("Number of target: ", len(bboxes))
    print("Input image size: ", img.shape)
        
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox.box()
        
        crop_img = img[y1:y2, x1:x2]
        
        # show_img(crop_img)
        
        text_regions = get_chars_region(crop_img)
        
        send_images = classify_char(crop_img, text_regions)
        return send_images
        pass
    
def classify_char(img, text_regions):
    image_transforms = { 
        'train': transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(size=64, scale=(0.8, 1.0)),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size=64),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size=(64, 64)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
        ])
    }
    
    send_images = []
    for region in text_regions:
        x, y, w, h = region
        crop_img = img[y:y+h, x:x+w] 
        crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
         
        # show_img(crop_img)
        
        # crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
        crop_img = image_transforms["test"](crop_img)
        send_images.append(crop_img.unsqueeze(axis=0))
        # crop_img = torch.unsqueeze(crop_img, axis=0)
        
        # results = model(crop_img) 
        # print(torch.argmax(results))
        
    send_images = torch.concat(send_images, axis=0)
        
    return send_images.cpu().detach().numpy()
    
    