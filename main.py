import streamlit as st
from PIL import Image
import numpy as np
import cv2
import functions
import util
import style

# Image config
widthImg = 600
heightImg = 780
marksPerQuestion = 1
choices = 4
questions = [20, 20, 20, 20, 20]  # Python, EDA, SQL, PowerBI, Statistics
THRESHOLDS = [50000, 30000, 10000]  # dynamic threshold
ANGLES = [0, 90, 180, 270]  # rotations for tilted sheets

# Answer key
raw_key = {
    "Python": ["a","c","c","c","c","a","c","c","b","c",
               "a","a","d","a","b","a,b,c,d","c","d","a","b"],
    "EDA": ["a","d","b","a","c","b","a","b","d","c",
            "c","a","b","c","a","b","d","b","a","b"],
    "SQL": ["c","c","c","b","b","a","c","b","d","a",
            "c","b","c","c","a","b","b","a","a,b","b"],
    "PowerBI": ["b","c","a","b","c","b","b","c","c","b",
                "b","b","d","b","a","b","b","b","b","b"],
    "Statistics": ["a","b","c","b","c","b","b","b","a","b",
                   "c","b","c","b","b","b","c","a","b","c"]
}

def parse_answer(ans_str):
    mapping = {'a':0,'b':1,'c':2,'d':3}
    ans_str = ans_str.replace(" ","").lower()
    if "," in ans_str:
        return [mapping[x] for x in ans_str.split(",")]
    else:
        return mapping[ans_str]

ans = [[parse_answer(x) for x in raw_key[sub]] for sub in raw_key]

# Dynamic contour detection
def get_biggest_contour(contours):
    for t in THRESHOLDS:
        rectCon = util.rectContours(contours, t)
        if len(rectCon) > 0:
            return rectCon
    return []

def find_marks(image, ans, questions):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (widthImg, heightImg))
    imgContours = img.copy()
    imgBiggestContours = img.copy()
    rectCon = []

    # Try rotated images
    for angle in ANGLES:
        rotated_img = functions.rotate_image(img, angle) if angle!=0 else img
        img1 = functions.preProcess(rotated_img)
        contours, _ = functions.findContours(img1, imgContours)
        rectCon = get_biggest_contour(contours)
        if len(rectCon)>0:
            img = rotated_img
            break

    st.write("Contours found:", len(rectCon))
    if len(rectCon)==0:
        st.image(img, caption="Preprocessed Sheet", use_container_width=True)
        return None

    biggestContour1 = util.getCornerPoints(rectCon[0])
    if biggestContour1.size==0:
        return None

    cv2.drawContours(imgBiggestContours, biggestContour1, -1, (0,255,0), 20)
    biggestContour1 = util.reorder(biggestContour1)

    pt1 = np.float32(biggestContour1)
    pt2 = np.float32([[0,0],[widthImg,0],[0,heightImg],[widthImg,heightImg]])
    matrix1 = cv2.getPerspectiveTransform(pt1, pt2)
    imgWrap = cv2.warpPerspective(img, matrix1, (widthImg,heightImg))

    h,w,_ = imgWrap.shape
    cut = (h*60)//100
    top = imgWrap[:cut,:]
    bottom = imgWrap[cut:,:]

    finalImage, subject_scores = functions.upper(
        top, bottom, imgContours,
        questions[0], choices, questions, ans, marksPerQuestion,
        return_scores=True
    )

    return finalImage, subject_scores

# Streamlit UI
st.set_page_config(page_title="OMR Sheet Evaluation System",
                   page_icon="📝",
                   layout="centered",
                   initial_sidebar_state="expanded")
style.apply_styling()

st.title("📝 Automated OMR Sheet Evaluation System")
st.write("Upload your OMR sheet image and click **Calculate Marks**.")

uploaded_file = st.file_uploader("Choose an OMR image...", type=["jpg","jpeg","png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded OMR Sheet', use_container_width=True)

    if st.button('Calculate Marks'):
        with st.spinner('Processing...'):
            result = find_marks(image, ans, questions)
            if result is not None:
                final_image, subject_scores = result
                st.image(final_image, caption='Graded OMR Sheet', use_container_width=True)

                total_score = sum(subject_scores)
                subjects = ["Python","EDA","SQL","PowerBI","Statistics"]
                st.write("### Score Summary:")
                for i, sub in enumerate(subjects):
                    st.write(f"{sub}: {subject_scores[i]}/{questions[i]}")
                st.write(f"**Total Score: {total_score}/100**")
            else:
                st.error("Could not detect the OMR sheet. Please upload a clearer image or try a different scan/angle.")
else:
    st.info("Please upload an OMR sheet image to continue.")
