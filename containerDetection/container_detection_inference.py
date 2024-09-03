import cv2
import numpy as np
import torch
from torchvision.ops import nms
from util.iou_extract import extract_candidates  
from model.dataset import preprocess_image
import os
from model.FRCNN import FRCNN


class ContainerDetectionInference:
    def __init__(self, model_path, device=None):
        """
        Initialize the ContainerDetectionInference class.

        Args:
            model_path (str): Path to the trained model.
            device (str): Device to run the model on ('cpu' or 'cuda'). If None, auto-detect.
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = FRCNN().to(self.device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()  # Set model to evaluation mode


    def _process_image(self, file_name):
        """
        Process a single image to detect objects.

        Args:
            file_name (str): Path to the image file.

        Returns:
            dict: A dictionary containing bounding boxes and rotation angle.
        """
        img = cv2.imread(file_name, 1)[..., ::-1]  # Read and convert color space
        img = cv2.resize(img, (244, 244))  # Resize image
        H, W, _ = img.shape

        candidates = extract_candidates(img)
        candidates = [(x, y, x + w, y + h) for x, y, w, h in candidates]

        input_image = preprocess_image(img / 255.)[None]  # Preprocess image
        rois = np.array([[x, y, X, Y] for x, y, X, Y in candidates])
        rois = rois / np.array([W, H, W, H])
        rixs = np.array([0] * len(rois))

        rois, rixs = [torch.Tensor(item).to(self.device) for item in [rois, rixs]]

        with torch.inference_mode():
            probs, thetas, deltas = self.model(input_image, rois, rixs)
            confs, clss = torch.max(probs, -1)

        confs, clss, probs, thetas, deltas = [tensor.detach().cpu().numpy() for tensor in [confs, clss, probs, thetas, deltas]]
        candidates = np.array(candidates)

        ixs = clss != 0
        confs, clss, probs, thetas, deltas, candidates = [tensor[ixs] for tensor in [confs, clss, probs, thetas, deltas, candidates]]
        bbs = candidates + deltas

        ixs = nms(torch.tensor(bbs.astype(np.float32)), torch.tensor(confs), 0.05)
        confs, clss, probs, thetas, deltas, candidates, bbs = [tensor[ixs] for tensor in [confs, clss, probs, thetas, deltas, candidates, bbs]]

        if len(ixs) == 1:
            confs, clss, probs, thetas, deltas, candidates, bbs = [tensor[None] for tensor in [confs, clss, probs, thetas, deltas, candidates, bbs]]

        if len(bbs) > 0:
            bbs = bbs[0] / np.array([W, H, W, H])
            return {'bbs': [float(x) for x in bbs], 'theta': float(round(thetas[0][0], 2))}
        else:
            return {'bbs': [], 'theta': 0}

    def _rotate_image(self,image, angle):
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result,rot_mat


    def _save_cropped_rotated_image(self, image_path, bbs, theta, output_path):
        """
        Save a cropped and rotated image based on bounding box and rotation angle.

        Args:
            image_path (str): Path to the input image.
            bbs (list): Bounding box coordinates [x1, y1, x2, y2].
            theta (float): Rotation angle in radians.
            output_path (str): Path to save the processed image.
        """

        print(image_path)
        img = cv2.imread(image_path)
        H, W, _ = img.shape

        if len(bbs) > 0:
            # Scale bounding boxes back to original image dimensions
            bbs = bbs * np.array([W, H, W, H])
            x1, y1, x2, y2 = bbs

            # Calculate center, width, and height of the bounding box
            w = x2 - x1
            h = y2 - y1
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2

            # Rotate image based on theta
            img, rotation_matrix = self._rotate_image(img, theta * (180 / np.pi))

            # Create a rotated rectangle and get its corner points
            rot_rectangle = ((x_center, y_center), (w, h), 0)
            box_points = cv2.boxPoints(rot_rectangle)
            box_points = np.int0(box_points)

            # Crop the rotated image
            x_min, x_max = min(box_points[:, 0]), max(box_points[:, 0])
            y_min, y_max = min(box_points[:, 1]), max(box_points[:, 1])
            img = img[y_min:y_max, x_min:x_max]

            # Resize image while maintaining aspect ratio
            img = self._resize_with_aspect_ratio(img, target_size=800)

            # Save the final image with padding
            padded_img = self._add_padding(img, target_size=800, padding_color=(255, 255, 255))
            cv2.imwrite(output_path, padded_img)


    def _resize_with_aspect_ratio(self, img, target_size):
        """
        Resize the image while maintaining the aspect ratio.

        Args:
            img (np.ndarray): Input image.
            target_size (int): Target size for the smaller dimension.

        Returns:
            np.ndarray: Resized image with the aspect ratio maintained.
        """
        old_h, old_w = img.shape[:2]
        scale = min(target_size / old_w, target_size / old_h)
        new_w, new_h = int(old_w * scale), int(old_h * scale)
        return cv2.resize(img, (new_w, new_h))


    def _add_padding(self, img, target_size, padding_color=(255, 255, 255)):
        """
        Add padding to an image to reach the target size.

        Args:
            img (np.ndarray): Input image.
            target_size (int): Target size for both width and height.
            padding_color (tuple): Color of the padding (BGR format).

        Returns:
            np.ndarray: Padded image of size (target_size, target_size).
        """
        old_h, old_w = img.shape[:2]
        pad_w = (target_size - old_w) // 2
        pad_h = (target_size - old_h) // 2
        result = np.full((target_size, target_size, 3), padding_color, dtype=np.uint8)
        result[pad_h:pad_h + old_h, pad_w:pad_w + old_w] = img
        return result


    def inference_model(self, files, output_path):
        """
        Run inference on a list of files.

        Args:
            files (list of str): List of paths to image files.

        Returns:
            list of dict: List of dictionaries containing bounding boxes and rotation angles for each image.
        """

        results = []        

        for file_path in files:
            file_name = os.path.basename(file_path)
            result = self._process_image(file_path)
            result['image_path']=str(file_path)

            if output_path != None and os.path.isdir(output_path):
                output_file = os.path.join(output_path, file_name)
                self._save_cropped_rotated_image(file_path,
                                                 result["bbs"],
                                                 result["theta"], 
                                                 output_file)
                result['output_file']=output_file
                
            
            results.append(result)
        return results
