import numpy as np
import sys
import json
import io

from enum import Enum

import cv2
# triton_python_backend_utils is available in every Triton Python model. You
# need to use this module to create inference requests and responses. It also
# contains some utility functions for extracting information from model_config
# and converting Triton input/output types to numpy types.
import triton_python_backend_utils as pb_utils
import torch
from PIL import Image
from skimage.filters import threshold_local

import torchvision.transforms as transforms
import os


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

class BoundingBox:
    def __init__(self, classID, confidence, x1, x2, y1, y2, image_width, image_height):
        self.classID = classID
        self.confidence = confidence
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.u1 = x1 / image_width
        self.u2 = x2 / image_width
        self.v1 = y1 / image_height
        self.v2 = y2 / image_height

    def box(self):
        return (self.x1, self.y1, self.x2, self.y2)

    def width(self):
        return self.x2 - self.x1

    def height(self):
        return self.y2 - self.y1

    def center_absolute(self):
        return (0.5 * (self.x1 + self.x2), 0.5 * (self.y1 + self.y2))

    def center_normalized(self):
        return (0.5 * (self.u1 + self.u2), 0.5 * (self.v1 + self.v2))

    def size_absolute(self):
        return (self.x2 - self.x1, self.y2 - self.y1)

    def size_normalized(self):
        return (self.u2 - self.u1, self.v2 - self.v1)
    
class LicensePlateTypes(Enum):
    TWO_LINE = 1
    ONE_LINE = 0

class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to intialize any state associated with this model.
        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """

        # You must parse model_config. JSON string is not parsed here
        self.model_config = model_config = json.loads(args['model_config'])

        self.output_name = "regions_output"
        # Get OUTPUT0 configuration
        output0_config = pb_utils.get_output_config_by_name(
            model_config, self.output_name)

        # Convert Triton types to numpy types
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config['data_type'])

    def execute(self, requests):
        """`execute` MUST be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference request is made
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse
        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest
        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """

        output0_dtype = self.output0_dtype

        responses = []

        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        for request in requests:
            # Get INPUT0
            in_0 = pb_utils.get_input_tensor_by_name(request, "INPUT_0")
            in_1 = pb_utils.get_input_tensor_by_name(request, "INPUT_1")
            in_2 = pb_utils.get_input_tensor_by_name(request, "INPUT_2")
            in_3 = pb_utils.get_input_tensor_by_name(request, "INPUT_3")
            in_4 = pb_utils.get_input_tensor_by_name(request, "INPUT_4")
            
            images = in_0.as_numpy()
            num_dets = in_1.as_numpy()
            det_boxes = in_2.as_numpy()
            det_scores = in_3.as_numpy()
            det_classes = in_4.as_numpy()
            
            image = self.preprocess_input_image(images[0])
            
            detected_objects = self.postprocess(num_dets, det_boxes, det_scores, det_classes, image.shape[1], image.shape[0], [416, 416])
            
            for idx in range(images.shape[0]):
                char_crop_images = self.get_text_regions(images[idx], detected_objects)
                print("Char crop: ", char_crop_images.shape, flush=True)
            
            out_tensor_0 = pb_utils.Tensor(self.output_name,
                                           char_crop_images.astype(output0_dtype))

            # Create InferenceResponse. You can set an error here in case
            # there was a problem with handling this inference request.
            # Below is an example of how you can set errors in inference
            # response:
            #
            # pb_utils.InferenceResponse(
            #    output_tensors=..., TritonError("An error occured"))
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor_0])
            responses.append(inference_response)

        # You should return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')
        
    
    def preprocess_input_image(self, image):
        image = image * 255
        # image = np.transpose(image, (0, 2, 3, 1)) # b, h, w, c
        
        return image
    
    def postprocess(self, num_dets, det_boxes, det_scores, det_classes, img_w, img_h, input_shape, letter_box=True):
        boxes = det_boxes[0, :num_dets[0][0]] / np.array([input_shape[0], input_shape[1], input_shape[0], input_shape[1]], dtype=np.float32)
        scores = det_scores[0, :num_dets[0][0]]
        classes = det_classes[0, :num_dets[0][0]].astype(np.int)

        old_h, old_w = img_h, img_w
        offset_h, offset_w = 0, 0
        if letter_box:
            if (img_w / input_shape[1]) >= (img_h / input_shape[0]):
                old_h = int(input_shape[0] * img_w / input_shape[1])
                offset_h = (old_h - img_h) // 2
            else:
                old_w = int(input_shape[1] * img_h / input_shape[0])
                offset_w = (old_w - img_w) // 2

        boxes = boxes * np.array([old_w, old_h, old_w, old_h], dtype=np.float32)
        if letter_box:
            boxes -= np.array([offset_w, offset_h, offset_w, offset_h], dtype=np.float32)
        boxes = boxes.astype(np.int)

        detected_objects = []
        for box, score, label in zip(boxes, scores, classes):
            detected_objects.append(BoundingBox(label, score, box[0], box[2], box[1], box[3], img_w, img_h))
        return detected_objects
    
    def preprocess(self, image):
        print("@@@")
        pass
        
    def get_text_regions(self, img, bboxes):
        print("Number of target: ", len(bboxes))
        print("Input image size: ", img.shape)
                
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox.box()
            print(f"Detected boxes: {bbox.box()}", flush=True)
            
            crop_img = img[y1:y2, x1:x2]
            
            # show_img(crop_img)
            text_regions = self.get_chars_region(crop_img)
            print("Regions: ", len(text_regions), flush=True)
            char_crop_images = self.get_chars_crop(crop_img, text_regions)
            break
            
        return char_crop_images
            
    def detect_license_plate_type(self, text_regions, ratio=0.8):
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
    
    def get_chars_region(self, crop_img):
        print(f"crop image size: {crop_img.shape}", flush=True)
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
            # area = stats[idx, cv2.CC_STAT_AREA]
            
            # filter candidates
            box_ratio = w / float(h)
            heigh_ratio = h / float(crop_img.shape[0])
            
            if(0.2 < box_ratio < 1 and 0.3 < heigh_ratio < 2):
                text_regions.append((x, y, w, h))
        
        lp_type = self.detect_license_plate_type(text_regions)
        
        if(lp_type == LicensePlateTypes.TWO_LINE):
            text_regions = sorted(text_regions, key=lambda x: x[1])
            text_regions = sorted(text_regions[:4], key=lambda x: x[0]) + sorted(text_regions[4:], key=lambda x: x[0])
            
        elif(lp_type == LicensePlateTypes.ONE_LINE):
            text_regions = sorted(text_regions, lambda x: x[0])
        
        return text_regions
    
    def get_chars_crop(self, img, text_regions):    
        send_images = []
        for region in text_regions:
            x, y, w, h = region
            crop_img = img[y:y+h, x:x+w] 
            crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB).astype(np.uint8)
            print("IN: ", crop_img.shape, flush=True)
            
            crop_img = image_transforms["test"](crop_img)
            send_images.append(crop_img.unsqueeze(axis=0))

        send_images = torch.concat(send_images, axis=0)
            
        return send_images.cpu().detach().numpy()