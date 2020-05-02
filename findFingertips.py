import cv2
import imutils
import math
import os
import numpy as np
from imutils import perspective
from imutils import contours
from sklearn.cluster import KMeans
from pandas import DataFrame


def show_images(images, prefix='prefix', image_name='image'):
    """
    Displays images on screen and saves them on the hard-drive.
    :param images: List of cv2 images to display and save.
    :param prefix: Image prefix, generally a description of what kind of transformation was applied to the image.
    :param image_name: Name of the current image being displayed, a new folder will be created for the image name supplied.
    :return: Nothing.
    """

    # Creates output folders if they do not exist.
    if not os.path.isdir(f'output'):
        os.mkdir(f'output')
    if not os.path.isdir(f'output/{image_name}'):
        os.mkdir(f'output/{image_name}')

    # For each image supplied
    for i, img in enumerate(images):
        # Reduce the size of the visualization if the image is too big.
        cv2.namedWindow(f'{prefix}_{i}')
        if img.shape[0] > 1000:
            tmp_img = cv2.resize(img, (int(img.shape[1] / 6), int(img.shape[0] / 6)))
        else:
            tmp_img = img
        # Organize the display windows on the screen for better visualization.
        cv2.moveWindow(f'{prefix}_{i}', i * int(tmp_img.shape[1] + 50) + 200, 0)
        cv2.imshow(f'{prefix}_{i}', tmp_img)
        # And finally save images on destination folder.
        if i != 0:
            cv2.imwrite(f'output/{image_name}/{prefix}_{str(i)}.jpg', img)
    # Display every image at the same time, if you want to visualize images one by one, move the following
    # couple of lines to the inside of the for loop above.
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def midpoint(ptA, ptB):
    """
    Simple support function that finds the arithmetic mean between two points.
    :param ptA: First point.
    :param ptB: Second point.
    :return: Midpoint.
    """
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


def crop_rect(img, rect, cut=True):
    """
    Crops a rectangle on an image and returns the rotated result.
    :param img: Input image to be cropped.
    :param rect: List of 4 points of the rectangle to crop the image.
    :param cut: Boolean to determine whether to actually cut the image before returning, or not.
    :return: Either a cropped image, or the full image with the rectangle drawn into it depending on the cut variable.
    """
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    if not cut:
        cv2.drawContours(img, [box], 0, (0, 255, 0), 10)

    W = rect[1][0]
    H = rect[1][1]

    Xs = [i[0] for i in box]
    Ys = [i[1] for i in box]
    x1 = min(Xs)
    x2 = max(Xs)
    y1 = min(Ys)
    y2 = max(Ys)

    rotated = False
    angle = rect[2]

    if angle < -45:
        angle += 90
        rotated = True

    mult = 1.0
    center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
    size = (int(mult * (x2 - x1)), int(mult * (y2 - y1)))

    if not cut:
        cv2.circle(img, center, 10, (0, 255, 0), -1)

    if not cut:
        size = (img.shape[0] + int(math.ceil(W)), img.shape[1] + int(math.ceil(H)))
        center = (int(size[0] / 2), int(size[1] / 2))

    M = cv2.getRotationMatrix2D((size[0] / 2, size[1] / 2), angle, 1.0)

    cropped = cv2.getRectSubPix(img, size, center)
    cropped = cv2.warpAffine(cropped, M, size)

    if cut:
        croppedW = W if not rotated else H
        croppedH = H if not rotated else W
    else:
        croppedW = img.shape[0] if not rotated else img.shape[1]
        croppedH = img.shape[1] if not rotated else img.shape[0]

    croppedRotated = cv2.getRectSubPix(cropped, (int(croppedW), int(croppedH)),
                                       (size[0] / 2, size[1] / 2))
    return croppedRotated


def getAngle(a, b, c):
    """
    Finds the angle (in degrees) between three points.
    :param a: First point.
    :param b: Second point.
    :param c: Third point.
    :return: A number between 0 and 360 degrees representing the angle between the three points.
    """
    ang = math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0]))
    return ang + 360 if ang < 0 else ang


def auto_canny(image, sigma=0.33):
    """
    Automatically finds the best params to detect edges on a gray image based on the median values of its pixels.
    :param image: Grayscale image.
    :param sigma: Hyper-parameter to determine how open or closed the threshold should be.
    (The lower the sigma, the higher the range).
    :return: Image containing the edges of the original image.
    """
    # Compute the median of the single channel pixel intensities
    v = np.median(image)

    # Apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))

    edged = cv2.Canny(image, lower, upper)

    return edged


if __name__ == "__main__":
    # Either run this script inside the images folder,
    # or change the folder string to the correct images folder location.
    folder = ''
    number_of_images = len(os.listdir(folder))

    for i, image_name in enumerate(os.listdir(folder)):
        biggestArea = 0
        biggestContour = None
        img = cv2.imread(f'{folder}/{image_name}')

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        done = False
        alpha = 2  # Contrast control (1.0-3.0)
        beta = 0  # Brightness control (0-100)
        while not done:
            contrast = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
            show_images([img, contrast], f'a_contrast', image_name)

            blur = cv2.bilateralFilter(contrast, 7, 150, 150)
            show_images([img, blur], f'b_blur', image_name)

            thresh = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY)[1]
            show_images([img, thresh], f'c_thresh', image_name)

            canny = auto_canny(thresh)
            show_images([img, canny], f'd_canny', image_name)

            kernel = np.array(([[0, 0, 1, 0, 0],
                                [0, 0, 1, 0, 0],
                                [1, 1, 1, 1, 1],
                                [0, 0, 1, 0, 0],
                                [0, 0, 1, 0, 0]]), dtype=np.uint8)
            eroded = cv2.dilate(canny, kernel, iterations=1)
            show_images([img, eroded], 'e_eroded', i)

            kernel = np.array(([[0, 0, 1, 0, 0],
                                [0, 0, 1, 0, 0],
                                [0, 0, 1, 0, 0],
                                [0, 0, 1, 0, 0],
                                [0, 0, 1, 0, 0]]), dtype=np.uint8)
            edged = cv2.dilate(eroded, kernel, iterations=1)
            show_images([img, edged], 'e_dilatedV', image_name)

            kernel = np.array(([[0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0],
                                [1, 1, 1, 1, 1],
                                [0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0]]), dtype=np.uint8)

            edged = cv2.dilate(edged, kernel, iterations=2)

            # Create lines on the image to encase fingers inside a box in order to generate a proper contour.
            cv2.line(edged, (0, edged.shape[0] - int(edged.shape[0] / 3)),
                     (edged.shape[1], edged.shape[0] - int(edged.shape[0] / 3)), (255, 0, 255), 10)
            cv2.rectangle(edged, (0, edged.shape[0] - int(edged.shape[0] / 3)), (edged.shape[1], edged.shape[0]),
                          (0, 0, 0), -1)
            cv2.rectangle(img, (0, edged.shape[0] - int(edged.shape[0] / 3)), (edged.shape[1], edged.shape[0]),
                          (0, 0, 0), -1)

            show_images([img, edged], 'f_dilatedH', image_name)

            kernel = np.ones((9, 9), np.uint8)
            closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel, iterations=1)
            show_images([img, closed], 'g_closed', image_name)

            cnts = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)

            # sort the contours from left-to-right
            (cnts, _) = contours.sort_contours(cnts)

            # Create a copy of the edged image in order to be able to perform more transformations in the image.
            orig = edged.copy()
            biggestContours = []
            for j, c in enumerate(cnts):
                # if the contour is not sufficiently large, ignore it
                area = cv2.contourArea(c)
                if area < ((orig.shape[0] * orig.shape[1]) / 10):
                    continue
                else:
                    biggestContours.append(c)

                # compute the rotated bounding box of the contour
                box = cv2.minAreaRect(c)
                box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
                box = np.array(box, dtype="int")

                # Order the points in the contour such that they appear in
                # top-left, top-right, bottom-right, and bottom-left order,
                # then draw the outline of the rotated bounding box
                box = perspective.order_points(box)
                cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 10)

                # loop over the original points and draw them
                for (x, y) in box:
                    cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

                # Unpack the ordered bounding box,
                (tl, tr, br, bl) = box

                # Then compute the midpoint between the top-left and top-right coordinates,
                (tltrX, tltrY) = midpoint(tl, tr)
                # Followed by the midpoint between bottom-left and bottom-right coordinates
                (blbrX, blbrY) = midpoint(bl, br)

                # Compute the midpoint between the top-left and top-right points,
                (tlblX, tlblY) = midpoint(tl, bl)
                # Followed by the midpoint between the top-right and bottom-right
                (trbrX, trbrY) = midpoint(tr, br)

                # Draw the midpoints on the image
                cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
                cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
                cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
                cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

                # Draw lines between the midpoints
                cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
                         (255, 0, 255), 2)
                cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
                         (255, 0, 255), 2)

                # Check if current contour has the biggest area in all of the contours, and save it if so.
                if area >= biggestArea:
                    biggestArea = area
                    biggestContour = c

            # If we found a contour, draw the biggest one on the image and display it.
            if len(biggestContours) > 0:
                contours_image = cv2.drawContours(img.copy(), biggestContours, -1, (255, 0, 0), 10)
                show_images([img, contours_image], 'h_contours', image_name)

            # Otherwise, reduce the alpha (lower the contrast on the image), and try again, until it is done,
            # or until we reach the original contrast of the image.
            if biggestArea != 0 or alpha <= 1:
                done = True
                print('done')
            else:
                alpha -= 0.5

        # If we still haven't found any contours on the image, print the image name and move on to the next one.
        if len(biggestContours) <= 0:
            print(f'Could not find contours on the image {image_name}.')
            continue

        # Select the bounding box of the biggest contour on the image (hopefully the hand).
        box = cv2.minAreaRect(biggestContour)

        show_images([img, orig], 'i_boxes', image_name)

        # Crop only the hand out of the original image, and display/save the transformed versions of it.
        hand = crop_rect(img, box)
        show_images([img, hand], f'j_hand', image_name)

        hand_gray = crop_rect(gray, box)
        hand_contrast = cv2.convertScaleAbs(hand_gray, alpha=alpha, beta=beta)
        show_images([hand, hand_contrast], f'k_hand_contrast', image_name)

        hand_blur = crop_rect(blur, box)
        show_images([hand, hand_blur], f'l_hand_blur', image_name)

        hand_canny = crop_rect(canny, box)
        show_images([hand, hand_canny], f'm_hand_canny', image_name)

        hand_edged = crop_rect(edged, box)
        show_images([hand, hand_edged], 'n_hand_dilated', image_name)

        mask = np.ones(img.shape[:2], dtype="uint8") * 255
        cv2.drawContours(mask, [biggestContour], -1, 0, -1)

        mask = (255 - mask)
        rotated = cv2.bitwise_and(img, img, mask=mask)
        final = rotated.copy()
        rotated = crop_rect(rotated, box)
        mask = crop_rect(mask, box)

        show_images([hand, rotated, mask], 'o_hand_final', image_name)

        # After we are able to separate only the hand, we can use the convex hull algorithm to find extreme points.
        hull = cv2.convexHull(biggestContour, returnPoints=True)

        xs = []
        ys = []
        for point in hull:
            # Don't select points on the very edged of the image, as they might be on the bounding boxes drawn into it.
            if point[0][0] > 5 and point[0][1] > 5 and img.shape[1] - 5 > point[0][0] < img.shape[0] - 5:
                xs.append(point[0][0])
                ys.append(point[0][1])
                Data = {'x': xs,'y': ys}

        # Since the algorithm returns many extreme points, we have to find the cluster centroids of the 5 fingers.
        df = DataFrame(Data, columns=['x', 'y'])
        if len(df) > 5:
            # To find the centroids, we use a simple KMeans unsupervised machine learning algorithm to group the points
            # into 5 distinct categories.
            kmeans = KMeans(n_clusters=5).fit(df)
            centroids = kmeans.cluster_centers_

            # Then we sort the centroids from left to right and draw circles on them.
            centroids = centroids[np.argsort(centroids[:, 0])]
            original_centroids = centroids.copy()
            fingertips = img.copy()
            for point in centroids:
                x = int(point[0])
                y = int(point[1])
                cv2.circle(fingertips, (x, y), int(final.shape[1] / 50), (0, 255, 255), -1)

            # Here we apply some heuristics to better approximate the center of the fingerprints,
            # Rather than the center of the tips of the fingers.
            # To account for the fact that we calculate the angle of the fingers from the center of the image,
            # We increase the x position of the first two centroids, and decrease it on the last two.
            for k, centroid in enumerate(centroids):
                if k == 0:
                    centroid[0] += int(final.shape[1] / 25)
                elif k == 1:
                    centroid[0] += int(final.shape[1] / 60)
                elif k == 3:
                    centroid[0] -= int(final.shape[1] / 60)
                elif k == 4:
                    centroid[0] -= int(final.shape[1] / 25)

            for point in centroids:
                x = int(point[0])
                y = int(point[1])
                cv2.circle(fingertips, (x, y), int(final.shape[1] / 50), (255, 255, 0), -1)

            # Now we use the better approximations of the centers of the fingertips, to find the leftmost
            # and rightmost edge of each finger, in order to calculate their proper midpoints.
            fingers = []
            for k, point in enumerate(centroids):
                x = int(point[0])
                y = int(point[1])

                if k == 0:
                    leftmost = 0
                else:
                    leftmost = max(0, int(centroids[k][0]) - int(final.shape[1] / 15))
                if k == 4:
                    rightmost = final.shape[1] - 15
                else:
                    rightmost = min(int(centroids[k + 1][0]) - int(final.shape[1] / 15), final.shape[1] - 15)

                top = 1
                bottom = min(y + int(final.shape[0] / 25), final.shape[0])
                y = bottom
                left = None
                right = None
                for l in range(rightmost - leftmost):
                    cur = l + leftmost
                    temp = list(final[y, cur, :])
                    if temp >= [5, 5, 5] and left is None and right is None:
                        left = cur
                    if temp < [5, 5, 5] and left is not None and right is None:
                        right = cur
                    elif l == rightmost - 1:
                        if left is None:
                            left = leftmost
                        if right is None:
                            right = rightmost

                if right is None or left is None:
                    fingers.append(((int(leftmost + (rightmost - leftmost) / 2), int(y)), rightmost - leftmost))
                else:
                    fingers.append(((int(left + (right - left) / 2), int(y)), right - left))


            if fingers[0][1] > fingers[4][1]:
                left = True
            else:
                left = False

            for m, finger in enumerate(fingers):
                cv2.circle(fingertips, (finger[0][0], finger[0][1]), int(final.shape[1] / 50), (255, 255, 255), -1)

                if left and m == 0:
                    finger_name = 'thumb'
                elif left and m == 1:
                    finger_name = 'index'
                elif left and m == 2:
                    finger_name = 'middle'
                elif left and m == 3:
                    finger_name = 'ring'
                elif left and m == 4:
                    finger_name = 'pinky'

                elif not left and m == 0:
                    finger_name = 'pinky'
                elif not left and m == 1:
                    finger_name = 'ring'
                elif not left and m == 2:
                    finger_name = 'middle'
                elif not left and m == 3:
                    finger_name = 'index'
                elif not left and m == 4:
                    finger_name = 'thumb'

                temp_finger = final.copy()
                half_width = int(finger[1] / 1.5)
                half_height = int(finger[1] / 1.5) * 2
                x_start = max(0, finger[0][0] - half_width)
                x_end = min(final.shape[1], finger[0][0] + half_width)
                y_start = max(finger[0][1] - half_height, 0)
                y_end = min(final.shape[0], finger[0][1] + half_height)
                temp_finger = temp_finger[y_start:y_end, x_start:x_end]
                show_images([hand, temp_finger], finger_name, image_name)

            fingertips = crop_rect(fingertips, box)
            show_images([hand, fingertips], 'p_hand_hull', image_name)
