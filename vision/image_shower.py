import cv2


def display_image_until_q(image, image_name):
    cv2.namedWindow(image_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(image_name, (800, 450))
    cv2.imshow(image_name, image)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyWindow(image_name)