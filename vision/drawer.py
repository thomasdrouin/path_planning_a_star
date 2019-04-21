import cv2
import numpy as np


def draw_contours(contours, display_image, color):
    draw_color = get_color_by_string(color)
    cv2.drawContours(display_image, contours, -1, draw_color, 1)


def draw_piece_name_from_piece_position(piece_position, display_image, color_or_form):
    center_x, center_y = piece_position
    name = color_or_form
    cv2.putText(display_image, name, (center_x - 20, center_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), 1)


def draw_point_on_image(position, sample_image, color="blanc"):
    point_color = get_color_by_string(color)
    cv2.circle(sample_image, position, 3, point_color, -1)


def draw_obstacle_top_contours(contours, display_image):
    for c in contours:
        (x, y), radius = cv2.minEnclosingCircle(c)
        center = (int(x), int(y))
        radius = int(radius)
        cv2.circle(display_image, center, radius, (0, 255, 0), 1)


def draw_obstacle_contour_from_center(centers, sample_image):
    for center in centers:
        radius = 50
        cv2.circle(sample_image, center, radius, (0, 255, 0), 1)


def draw_start_squares_from_contours(contours, sample_image):
    for c in contours:
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        cv2.drawContours(sample_image, [c], -1, (0, 255, 0), 1)

        name = "Start square"
        cv2.putText(sample_image, name, (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), 1)


def draw_squares_from_contours(contour, sample_image):
    rect = cv2.minAreaRect(contour[0])
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(sample_image, [box], 0, (0, 0, 255), 2)


def get_color_by_string(color_string):
    color = (255, 255, 255)
    if color_string == "bleu":
        color = (255, 0, 0)
    elif color_string == "rouge":
        color = (0, 0, 255)
    elif color_string == "vert":
        color = (0, 255, 0)
    elif color_string == "jaune":
        color = (0, 255, 255)
    return color
