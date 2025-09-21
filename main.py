import streamlit as st
from PIL import Image
import numpy as np
import cv2
import functions
import util
import style

widthImg = 600
heightImg = 780
marksPerQuestion = 1   # each question = 1 mark
choices = 4            # a,b,c,d
questions = [20, 20, 20, 20, 20]   # Python, EDA, SQL, Power BI, Statistics


def parse_answer(ans_str):
    """Convert 'a,b,c' -> [0,1,2] and 'a' -> 0"""
    mapping = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
    ans_str = ans_str.replace(" ", "").lower()
    if "," in ans_str:
        return [mapping[x] for x in ans_str.split(",")]
    else:
        return mapping[ans_str]

# Build full answer key (100 Qs → 5 subjects)
raw_key = {
    "Python": [
        "a","c","c","c","c","a","c","c","b","c",
        "a","a","d","a","b","a,b,c,d","c","d","a","b"
    ],
    "EDA": [
        "a","d","b","a","c","b","a","b","d","c",
        "c","a","b","c","a","b","d","b","a","b"
    ],
    "SQL": [
        "c","c","c","b","b","a","c","b","d","a",
        "c","b","c","c","a","b","b","a","a,b","b"
    ],
    "PowerBI": [
        "b","c","a","b","c","b","b","c","c","b",
        "b","b","d","b","a","b","b","b","b","b"
    ],
    "Statistics": [
        "a","b","c","b","c","b","b","b","a","b",
        "c","b","c","b","b","b","c","a","b","c"
    ]
}

ans = [[parse_answer(x) for x in raw_key[sub]] for sub in raw_key]


def find_marks(image, ans, questions):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (widthImg, heightImg))
    imgContours = img.copy()
    imgBiggestContours = img.copy()

    img1 = functions.preProcess(img)
    contours, hierarchy = functions.findContours(img1, imgContours)

    rectCon = util.rectContours(contours, 200000)

    # ✅ Safe check for empty contours
    if len(rectCon) > 0:
        biggestContour1 = util.getCornerPoints(rectCon[0])

        if biggestContour1.size != 0:
            cv2.drawContours(imgBiggestContours, biggestContour1, -1, (0, 255, 0), 20)
            biggestContour1 = util.reorder(biggestContour1)

            pt1 = np.float32(biggestContour1)
            pt2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
            matrix1 = cv2.getPerspectiveTransform(pt1, pt2)
            imgWrap = cv2.warpPerspective(img, matrix1, (widthImg, heightImg))

            h, w, channels = imgWrap.shape
            cut = (h * 60) // 100

            top = imgWrap[:cut, :]
            bottom = imgWrap[cut:, :]

            finalImage = functions.upper(
                top, bottom, imgContours,
                questions[0], choices, questions, ans, marksPerQuestion
            )
            return finalImage
        else:
            return None
    else:
        # ❌ Return None if no contours found
        return None


st.set_page_config(page_title="OMR Sheet Evaluation System", page_icon="📝", layout="centered", initial_sidebar_state="expanded")
style.apply_styling()

st.title("📝 Automated OMR Sheet Evaluation System")
st.write("Upload your OMR sheet image below and click **Calculate Marks** to get results.")

uploaded_file = st.file_uploader("Choose an OMR image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded OMR Sheet', use_column_width=True, width=300)

    if st.button('Calculate Marks'):
        with st.spinner('Processing...'):
            final_image = find_marks(image, ans, questions)
            if final_image is not None:
                st.image(final_image, caption='Graded OMR Sheet', use_column_width=True)
            else:
                st.error("Could not detect the OMR sheet. Please upload a clearer image.")
else:
    st.info("Please upload an OMR sheet image to continue.")
