import json
import os.path
import numpy as np
import random

class COCOProcessor:
    def __init__(self, image_path, orig_coco_path, annotation_path):
        """
        Initialize the COCOProcessor class.
        
        Parameters:
        image_path (str): Path to the directory containing the dataset images.
        orig_coco_path (str): Path to the JSON file containing COCO annotations.
        annotation_path (str): Path to the JSON file containing processed annotations.
        test_ratio (float): Ratio of images to be used for validation. Defaults to 0.3 (30%).
        """
        self.image_path = image_path
        self.annotation_path = annotation_path
        self.coco_path = orig_coco_path       
        self.coco_json = self._load_annotations()
        self.annotations_file = {'images': [], 'annotations': []}
        self.all_corners = []

    def _load_annotations(self):
        """Load the COCO annotation JSON file."""
        with open(self.coco_path, 'r') as f:
            return json.load(f)
    
    def _split_ids(self):
        """Split image IDs into training and validation sets based on the test ratio."""
        ids = [x['id'] for x in self.coco_json['images']]
        test_ids = random.sample(ids, int(len(ids) * self.test_ratio))
        train_ids = list(set(ids) - set(test_ids))
        return train_ids, test_ids
    
    def _calc_bearing(self, corner1, corner2):
        """Calculate the rotation angle (bearing) from two corner points."""
        dx = corner2[0] - corner1[0]
        dy = corner2[1] - corner1[1]
        return round(np.arctan2(dy, dx), 2)

    def _segmentation_corners_to_bbox(self, corners):

        # Calculate the center of the corners
        centre = np.mean(np.array(corners), axis=0)
        
        # Calculate the minimum and maximum x, y coordinates to get the axis-aligned bounding box
        x_min = np.min(corners[:, 0])
        y_min = np.min(corners[:, 1])
        x_max = np.max(corners[:, 0])
        y_max = np.max(corners[:, 1])
        
        # Calculate width and height of the bounding box
        width = x_max - x_min
        height = y_max - y_min
        
        # Return bounding box in the format [x_min, y_min, width, height]
        return [x_min, y_min, x_max, y_max]
    
    def _segmentation_corners_to_rotated_bbox(self, corners):
        """Convert segmentation corners to a rotated bounding box."""
        centre = np.mean(np.array(corners), 0)
        theta = self._calc_bearing(corners[0], corners[1])
        rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        out_points = np.matmul(corners - centre, rotation) + centre
        x, y = list(out_points[0, :])
        w, h = list(out_points[2, :] - out_points[0, :])
        return [x, y, abs(w), abs(h), theta]
    
    def _segmentation_to_corners(self, segmentation, img_width, img_height):
        """Convert segmentation data into corner coordinates."""
        corners = [[segmentation[x] * img_width, segmentation[x + 1] * img_height]
                   for x in range(0, len(segmentation), 2)]

        # Remove duplicates
        temp = []
        for x in corners:
            if x not in temp:
                temp.append(x)
        corners = temp

        # Find the center and sort corners accordingly
        centre = np.mean(np.array(corners), 0)
        for i in range(len(corners)):
            if corners[i][0] < centre[0]:
                if corners[i][1] < centre[1]:
                    corners[i], corners[0] = corners[0], corners[i]
                else:
                    corners[i], corners[3] = corners[3], corners[i]
            else:
                if corners[i][1] < centre[1]:
                    corners[i], corners[1] = corners[1], corners[i]
                else:
                    corners[i], corners[2] = corners[2], corners[i]

        return corners

    def _bbox_from_list(self, bbox, img_width, img_height):
        """Convert a normalized bounding box into corner coordinates."""
        x = bbox[0] * img_width
        y = bbox[1] * img_height
        w = bbox[2] * img_width
        h = bbox[3] * img_height
        corners = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
        c_x, c_y = np.mean(np.array(corners), 0)
        return [c_x, c_y, w, h]
    
    def _img_exists(self, file_name):
        return os.path.isfile(os.path.join(self.image_path, file_name))

    def process_annotations(self):
        """Process each annotation, converting segmentation and bounding boxes."""
        for index, annotation_json in enumerate(self.coco_json['annotations']):
            image_id = annotation_json['image_id']
            image_json = self.coco_json['images'][image_id - 1]
            if self._img_exists(image_json['file_name']):
                segmentation = annotation_json['segmentation'][0]
                img_width = image_json['width']
                img_height = image_json['height']

                 # Convert segmentation to corners and process bounding boxes
                corners = self._segmentation_to_corners(segmentation, img_width, img_height)                
                bbox = annotation_json['bbox']

                s_x, s_y, w, h, theta = self._segmentation_corners_to_rotated_bbox(corners)
                x, y, b_w, b_h = self._bbox_from_list(bbox, img_width, img_height)

                annotation_json['segmentation'] = [c for sublist in corners for c in sublist]
                annotation_json['bbox'] = [(x - w / 2), (y - h / 2), (x + w / 2), (y + h / 2)]

                self.annotations_file['images'].append(image_json)
                self.annotations_file['annotations'].append(annotation_json)

    def save_to_file(self):
        """Save processed annotation files."""
        with open(self.annotation_path, 'w') as f:
            json.dump(self.annotations_file, f)

    def shuffle_annotations(self):
        """Shuffle the order of the annotations and images."""
        indexes = list(range(0, len(self.annotations_file['images'])))
        random.shuffle(indexes)

        for i, j in enumerate(indexes):
            self.annotations_file['images'][i], self.annotations_file['images'][j] = self.annotations_file['images'][j], self.annotations_file['images'][i]
            self.annotations_file['annotations'][i], self.annotations_file['annotations'][j] = self.annotations_file['annotations'][j], self.annotations_file['annotations'][i]
            #self.all_corners[i], self.all_corners[j] = self.all_corners[j], self.all_corners[i]




