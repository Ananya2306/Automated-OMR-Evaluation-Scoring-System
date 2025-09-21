import cv2
import numpy as np

def preProcess(img):
    """Convert image to grayscale and apply blur + adaptive threshold"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 1)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 25, 10)
    return thresh

def findContours(imgThresh, imgContour):
    """Find external contours and return them"""
    contours, hierarchy = cv2.findContours(imgThresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours, hierarchy

def rotate_image(img, angle):
    """Rotate image by 0,90,180,270 degrees"""
    if angle == 0:
        return img
    elif angle == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        return img

def upper(top, bottom, imgContours, questions_per_subject, choices, questions, ans, marksPerQuestion, return_scores=False):
    """
    Process top and bottom halves of OMR sheet,
    detect filled bubbles and calculate marks.
    Returns:
        final graded image
        (optional) subject_scores list
    """
    # Combine top and bottom
    img = np.vstack([top, bottom])
    h, w, _ = img.shape
    subject_scores = []

    # Approx rows division (each subject block)
    start_row = 0
    for i, q_count in enumerate(questions):
        block_height = h // len(questions)  # rough division
        sub_img = img[start_row:start_row+block_height, :]
        start_row += block_height

        # Divide into rows and columns for bubble detection
        row_height = block_height // q_count
        col_width = w // choices

        score = 0
        for q in range(q_count):
            for c in range(choices):
                x = c * col_width
                y = q * row_height
                cell = sub_img[y:y+row_height, x:x+col_width]
                total_pixels = cv2.countNonZero(cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY))
                if isinstance(ans[i][q], list):
                    if c in ans[i][q] and total_pixels > (row_height*col_width)//4:
                        score += marksPerQuestion
                        cv2.rectangle(img, (x, y+start_row-block_height), (x+col_width, y+row_height+start_row-block_height), (0,255,0), 2)
                else:
                    if c == ans[i][q] and total_pixels > (row_height*col_width)//4:
                        score += marksPerQuestion
                        cv2.rectangle(img, (x, y+start_row-block_height), (x+col_width, y+row_height+start_row-block_height), (0,255,0), 2)
        subject_scores.append(score)

    if return_scores:
        return img, subject_scores
    else:
        return img
