import cv2
import numpy as np
import argparse


# python segment_visualize.py --input /home/EricHu/Workspace/argoverse/argoverse-tracking/argo_track/val/instance_segment/0000/000045.png

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

    masked_image = cv2.bitwise_and(image, mask) * 7 // 8 + image // 8
    return masked_image 

def get_crop_area_from_color(src):
    # Get tid color
    src_copy = src[np.invert(
                    np.all(src == (0.0, 0.0, 0.0), axis=-1) | \
                    np.all(src == (255, 255, 255), axis=-1))
                    ]
    tid_color = np.unique(src_copy.reshape(-1, 3), axis=0)

    # Get crop area
    key = 1
    for tcolor in tid_color:
        print(tcolor)
        # Get array points from color
        tpoints = np.argwhere(np.all(src==tcolor, axis=-1))

        # Then, crop the segmentation area using convex hull
        crop_out = crop_out_polygon_convex(rgb, tpoints)

        # Show in a window
        while(1):
            crop_out[tpoints[:, 0], tpoints[:, 1]] = (255, 255, 255)
            cv2.imshow('Contours', cv2.resize(crop_out, (0, 0), fx=0.5, fy=0.5))
            key = cv2.waitKey(1)
            if key in [27, ord(' ')]:
                break
        if key == 27:
            cv2.destroyAllWindows()
            exit()

if __name__ == '__main__':
    # Load source image
    parser = argparse.ArgumentParser(description='Code for Convex Hull tutorial.')
    parser.add_argument('--input', help='Path to input image.', default='stuff.jpg')
    args = parser.parse_args()

    src = cv2.imread(cv2.samples.findFile(args.input))
    assert src is not None, f"Could not open or find the image: {args.input}"
    rgb = cv2.imread(cv2.samples.findFile(args.input.replace('instance_segment', 'image_02')))
    assert rgb is not None, f"Could not open or find the image: {args.input.replace('instance_segment', 'image_02')}"

    # Create Window
    #source_window = 'Source'
    #cv2.namedWindow(source_window)
    #cv2.imshow(source_window, cv2.resize(rgb//2 + src//2, (0, 0), fx=0.5, fy=0.5))

    get_crop_area_from_color(src)

    cv2.destroyAllWindows()
