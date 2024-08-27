import numpy as np



# Calculate rotation from max/min segmentation corners
def calc_bearing(corner1, corner2):

    # Difference in x coordinates
    dx = corner2[0] - corner1[0]

    # Difference in y coordinates
    dy = corner2[1] - corner1[1]

    theta = round(np.arctan2(dy, dx), 2)
    return theta



def segmentationCorners2rotatedbbox(corners):
    centre = np.mean(np.array(corners), 0)
    theta = calc_bearing(corners[0], corners[1])
    rotation = np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta), np.cos(theta)]])
    out_points = np.matmul(corners - centre, rotation) + \
        centre  
    x, y = list(out_points[0, :])
    w, h = list(out_points[2, :] - out_points[0, :])
    return [x, y, w, h, theta]




def segmentationToCorners(segmentation, img_width, img_height):
    corners = [[segmentation[x]*img_width, segmentation[x+1]*img_height]
               for x in range(0, len(segmentation), 2)]

    temp = []

    for x in corners:
        if x not in temp:
            temp.append(x)

    corners = temp

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


def bboxFromList(bbox, img_width, img_height):
    x = bbox[0] * img_width
    y = bbox[1] * img_height
    w = bbox[2] * img_width
    h = bbox[3] * img_height

    corners = [[x, y], [x+w, y], [x+w, y+h], [x, y+h]]
    c_x, c_y = np.mean(np.array(corners),0)

    return [c_x, c_y, w, h]