"""The code to translate Argoverse dataset to COCO dataset format"""

import os
import cv2
import json
import argparse
import numpy as np

from skimage import measure
from shapely.geometry import Polygon, MultiPolygon


# python adapter.py val

def parse_args():
    parser = argparse.ArgumentParser(description='Convert data format from Argoverse to KITTI',
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('data_split', choices=['train', 'val', 'test', 'sample'], 
                        help='Which data split to use in testing')
    parser.add_argument('--dry_run', action='store_true', default=False,
                        help='Show command without running')
    parser.add_argument('--vis', action='store_true', default=False,
                        help='Show label')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='Print the output logs')
    parser.add_argument('--vvv', action='store_true', default=False,
                        help='Print all output logs')
    
    args = parser.parse_args()

    return args


args = parse_args()

def crop_out_polygon_convex(image: np.ndarray, 
                            point_array: np.ndarray,
                            ignore_mask_color: tuple = (255, 255, 255)
                            ) -> np.ndarray:
    """
    Crops out a convex polygon given from a list of points from an image
    :param image: Opencv BGR image
    :param point_array: list of points that defines a convex polygon
    :param ignore_mask_color: tuple of color that ignores in fillConverPoly 
    :return: Cropped out image
    """
    mask = np.zeros(image.shape, dtype=np.uint8)

    roi_corners = cv2.convexHull(point_array).squeeze()
 
    # Flip from (row, col) representation to (x, y)
    if roi_corners.ndim == 2:
        roi_corners = roi_corners[:, ::-1]
    else:
        roi_corners = roi_corners.reshape(-1, 2)[:, ::-1]

    cv2.fillConvexPoly(mask, roi_corners, ignore_mask_color)

    return mask
    # NOTE: If we use points instead
    #masked_image = cv2.bitwise_and(image, mask)
    #return masked_image 

def create_sub_masks(mask_image: np.ndarray) -> np.ndarray:
    """
    Crops out convex polygons given from a list of points from an image
    :param mask_image: Opencv BGR image
    :return: Cropped out image
    """
    # Initialize a dictionary of sub-masks indexed by RGB colors
    sub_masks = {}

    # Get tid color
    src_copy = mask_image[np.invert(
                    np.all(mask_image == (0.0, 0.0, 0.0), axis=-1) | \
                    np.all(mask_image == (255, 255, 255), axis=-1))
                    ]
    if len(src_copy) == 0:
        if args.verbose: print("No object in this frame.")
        return sub_masks

    tid_color = np.unique(src_copy.reshape(-1, 3), axis=0)

    # Convert image to gray and blur it
    mask_image_gray = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)

    # Get crop area
    for tcolor in tid_color:
        # Get array points from color
        tpoints = np.argwhere(np.all(mask_image==tcolor, axis=-1))

        # Then, crop the segmentation area using convex hull
        crop_out = crop_out_polygon_convex(mask_image_gray, tpoints)
        sub_masks[tuple(tcolor)] = crop_out

        # Show in a window
        if args.vis:
            crop_out[tpoints[:, 0], tpoints[:, 1]] = 255
            cv2.imshow('Contours', cv2.resize(crop_out, (0, 0), fx=0.5, fy=0.5))
            cv2.waitKey(0)

    return sub_masks

def create_sub_mask_annotation(sub_mask: np.ndarray, 
                               image_id: int, 
                               category_id: int, 
                               annotation_id: int, 
                               is_crowd: bool) -> dict:
    """
    Find contours (boundary lines) around each sub-mask
    Note: there could be multiple contours if the object
    is partially occluded. (E.g. an elephant behind a tree)
    :param sub_mask: array of points that masks an object 
    :param image_id: int vlaue for image index
    :param category_id: int value for category index 
    :param annotation_id: int value for annotation index 
    :param is_crowd: bool indicates if the object is crowded with others 
    :return: annotation dict 
    """
    contours = measure.find_contours(sub_mask, 0.5, positive_orientation='low')

    segmentations = []
    polygons = []
    for contour in contours:
        # Make a polygon and simplify it
        poly = Polygon(contour)
        poly = poly.simplify(0.1, preserve_topology=True)
        if poly.is_empty:
            if args.vvv: print("Skip poly")
            continue
        else:
            polygons.append(poly)
            segmentation = np.array(poly.exterior.coords).ravel().tolist()
            segmentations.append(segmentation)

    # Combine the polygons to calculate the bounding box and area
    multi_poly = MultiPolygon(polygons)
    if multi_poly.is_empty:
        if args.vvv: print("Skip multipoly")
        x, y = 0, 0
        width = 0
        height = 0
        bbox = (x, y, width, height)
        area = 0
    else:
        x, y, max_x, max_y = multi_poly.bounds
        width = max_x - x
        height = max_y - y
        bbox = (x, y, width, height)
        area = multi_poly.area

    annotation = {
        'segmentation': segmentations,
        'iscrowd': is_crowd,
        'image_id': image_id,
        'category_id': category_id,
        'id': annotation_id,
        'bbox': bbox,
        'area': area
    }

    return annotation

def get_tid_from_color(color: (int, int, int)) -> int:
    # NOTE: the color is in RGB format
    #       and COCO treats (0, 0, 0) as ignore, tid cannot be 0
    return int((color[0] << 16) + (color[1] << 8) + color[2]) - 1

def main():
    # Root directory
    ROOT_DIR = '/home/EricHu/Workspace/argoverse/argoverse-tracking/'
    TARGET_DIR = f'/home/EricHu/Workspace/detectron2/datasets/argoverse/annotations/'

    # Setup directories
    goal_dir = os.path.join(ROOT_DIR, 'argo_track', args.data_split)
    inst_seg_dir = os.path.join(goal_dir, 'instance_segment')
    image_dir = os.path.join(goal_dir, 'image_02')
    out_json_name = os.path.join(TARGET_DIR, f'argo_{args.data_split}.json')

    # Define which colors match which categories in the images
    category_id = 3
    is_crowd = 0

    # These ids will be automatically increased as we go
    image_id = 0

    # Create the annotations
    attr_dict = {}
    annotations = []
    images = []

    for dir_name in sorted(os.listdir(inst_seg_dir)):
        inst_seg_vid_dir = os.path.join(inst_seg_dir, dir_name)
        image_vid_dir = os.path.join(image_dir, dir_name)
        print(f"Video {inst_seg_vid_dir}")

        for img_name in sorted(os.listdir(inst_seg_vid_dir)):
            inst_seg_img_name = os.path.join(inst_seg_vid_dir, img_name)
            image_name = os.path.join(image_vid_dir, img_name)
            image_id += 1
            print(f"Segment image {inst_seg_img_name}")

            image_dict = {'file_name': image_name, 
                     'height': 1200, 
                     'width': 1920, 
                     'id': image_id}

            # NOTE: Get image with transposed height and width (like in Image)
            #       Contours work properly with images loaded with skimage, 
            #       but not with OpenCV.
            image = cv2.imread(inst_seg_img_name).transpose(1, 0, 2)
            
            sub_masks = create_sub_masks(image)
            for color, sub_mask in sub_masks.items():
                print(f"Image {image_id} - ID {get_tid_from_color(color)} with Color {color}")
                annotation = create_sub_mask_annotation(sub_mask, 
                                                        image_id, 
                                                        category_id, 
                                                        get_tid_from_color(color), 
                                                        is_crowd)
                if annotation['area']:
                    annotations.append(annotation)
                    if args.verbose:
                        print(json.dumps(annotation))

            images.append(image_dict)


    attr_dict['images'] = images
    attr_dict['annotations'] = annotations
    attr_dict['type'] = 'instances'
    attr_dict['categories'] = [
        {"supercategory": "vehicle", "id": 3, "name": "car"},
        ]
    if args.dry_run:
        print(f"Dump to screen")
        print(json.dumps(attr_dict))
    else:
        print(f"Dump to {out_json_name}")
        with open(out_json_name, 'w') as f:
            json.dump(attr_dict, f) 

if __name__ == '__main__':
    main()
