import streamlit as st
import mysql.connector
import pandas as pd
import csv
import os
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io
from reportlab.lib.pagesizes import letter
import io
from PIL import Image
from reportlab.lib.units import inch
import tempfile
from reportlab.lib import colors
import cv2
from ultralytics import YOLO



def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="acc_information"
    )

def authenticate(username, password):
    conn = get_db_connection()
    cursor = conn.cursor()
    query = "SELECT * FROM users WHERE username=%s AND password=%s"
    cursor.execute(query, (username, password))
    result = cursor.fetchone()
    conn.close()
    return result is not None

def signup_user(username, password, email, Age, Address):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO users (username, password, email, age, address) VALUES (%s, %s, %s, %s, %s)", (username, password, email, Age, Address))
        conn.commit()
        return True
    except mysql.connector.IntegrityError:
        return False
    finally:
        conn.close()

# -------------------- Styling --------------------
def set_background():
    st.markdown(
        """
        <style>
        .stApp::before {
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            height: 100%;
            width: 100%;
            background-image: url("https://static.vecteezy.com/system/resources/thumbnails/036/372/442/small_2x/hospital-building-with-ambulance-emergency-car-on-cityscape-background-cartoon-illustration-vector.jpg");
            background-size: cover;
            background-position: center;
            filter: blur(22px);
            z-index: -1;
        }
        .stApp {
            background: transparent;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


def set_full_page_height():
    st.markdown("""
        <style>
        .main {
            min-height: 100vh;  /* Full viewport height */
        }
        </style>
    """, unsafe_allow_html=True)


# -------------------- Data Loading --------------------

def profile_dropdown():
    with st.sidebar:
        st.markdown(f"## 👤 {st.session_state['current_user']}")
        # st.markdown(f"**User:** {st.session_state['current_user']}")
        if st.button("🏠 Home"):
            st.session_state['page'] = 'home'
            st.rerun()
        # if st.button("📜 History"):S
        #     st.success("Showing your watch history... (Coming soon)")
        if st.button("🚪 Logout"):
            st.session_state["logged_in"] = False
            st.rerun()

def get_user_id(username):
    conn = get_db_connection()
    cursor = conn.cursor()
    query = "SELECT id FROM users WHERE username = %s"
    cursor.execute(query, (username,))
    result = cursor.fetchone()
    conn.close()
    if result:
        return result[0]  # Returns the 'id' value
    return None


# -------------------- Authentication Page --------------------
def auth_page():
    set_background()
    set_full_page_height()
    st.markdown("<h1 style='text-align: center; color: black;'> Green Medical Camp</h1>", unsafe_allow_html=True)

    # Make the entire tab content live inside a bordered box
    st.markdown("""
        <style>
        div[data-testid="stTabs"] div[role="tabpanel"]{
            border: 1px solid #000000;
            border-radius: 12px;
            padding: 24px;
            margin-top: 12px;
            background: rgba(255,255,255,0.65);
        }
        </style>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs([" Login ", " Sign Up "])

    # Make tab labels bold
    st.markdown("""
        <style>
        button[data-baseweb="tab"] div p {
            font-weight: 700 !important;  /* bold */
        }
        </style>
    """, unsafe_allow_html=True)

    with tab1:
        with st.form("login_form"):
            st.text_input("Username", key="login_user")
            st.text_input("Password", type="password", key="login_pass")
            if st.form_submit_button("Login"):
                if authenticate(st.session_state.login_user, st.session_state.login_pass):
                    st.session_state["logged_in"] = True
                    st.session_state["current_user"] = st.session_state.login_user
                    st.session_state["user_id"] = get_user_id(st.session_state.login_user)
                    st.session_state['page'] = 'home'
                    st.rerun()
                else:
                    st.error("Invalid username or password")

    with tab2:
        with st.form("signup_form"):
            st.text_input("Username", key="signup_user")
            st.text_input("Password", type="password", key="signup_pass")
            st.text_input("Age", key="signup_Age")
            st.text_input("Address", key="signup_Address")
            st.text_input("Email", key="signup_mail")

            if st.form_submit_button("Sign Up"):
                u = st.session_state.signup_user.strip()
                p = st.session_state.signup_pass.strip()
                g = st.session_state.signup_Age.strip()
                f = st.session_state.signup_Address.strip()
                e = st.session_state.signup_mail.strip()

                if not u or not p or not e or not f or not g:
                    st.error("All fields are required.")
                elif "@" not in e or not e.endswith(".com"):
                    st.error("Enter a valid email address")
                else:
                    if signup_user(u, p, e, g, f):
                        st.success("Account created! You can now log in.")
                    else:
                        st.warning("Username already exists. Try another.")


def fetch_user_info(user_id):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT username, email, age, address FROM users WHERE id = %s", (user_id,))
    user_info = cursor.fetchone()
    cursor.close()
    conn.close()
    return user_info or {}


def generate_pdf(user_id, prediction, confidence, pil_img):
    # fetch user info from DB
    if confidence > 0.5:
        prediction = "Tumor Detected"
        pred_color = colors.red
    else:
        prediction = "Tumor not Detected"
        pred_color = colors.green

    user_info = fetch_user_info(user_id)

    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # ----- Header -----
    c.setFont("Helvetica-Bold", 24)
    c.drawCentredString(width / 2, height - 50, "Green Medical Camp")
    c.setFont("Helvetica-Bold", 18)
    c.drawCentredString(width / 2, height - 80, "Brain Tumor Detection Report")

    line_y = height - 90
    c.setLineWidth(1)
    c.line(50, line_y, width - 50, line_y)

    # ----- Image -----
    y_start = height - 95
    if pil_img is not None:  #
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            pil_img.save(tmp_file.name, format="PNG")
            image_width = 4 * inch
            image_height = 4 * inch
            # Position image in the middle horizontally
            img_x = width / 2 - (image_width / 2)
            img_y = y_start - image_height
            c.drawImage(tmp_file.name, img_x, img_y, width=image_width, height=image_height)
        # Move y_start below image for next content
        y_start = img_y - 40

    # ----- Patient Info (below picture) -----
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y_start, "Patient Information:")

    c.setFont("Helvetica", 12)
    c.drawString(60, y_start - 20, f"Name: {user_info.get('username', 'N/A')}")
    c.drawString(60, y_start - 40, f"Age: {user_info.get('age', 'N/A')}")
    c.drawString(60, y_start - 60, f"Address: {user_info.get('address', 'N/A')}")
    c.drawString(60, y_start - 80, f"Email: {user_info.get('email', 'N/A')}")

    # ----- Model Info (after patient info) -----
    model_y = y_start - 110
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, model_y, "Model Information:")

    c.setFont("Helvetica", 12)
    c.drawString(60, model_y - 20, "Model: YOLO-v8 2024")
    c.drawString(60, model_y - 40, "Parameter: 3,011,043")
    c.drawString(60, model_y - 60, "layers: 129")
    c.drawString(60, model_y - 80, "Accuracy: 96%")
    c.drawString(60, model_y - 105, f"Generated At: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ----- Horizontal line -----
    final_line_y = model_y - 120
    c.setLineWidth(1)
    c.line(50, final_line_y, width - 50, final_line_y)

    # ----- Final Prediction (at the bottom) -----
    c.setFont("Helvetica-Bold", 18)  # bigger font
    c.setFillColor(pred_color)
    c.drawString(50, final_line_y - 30, f"{prediction}")
    c.setFillColor(colors.black)
    # ----- Finish PDF -----
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer




# -------------------- Main Interface --------------------
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if not st.session_state["logged_in"]:
    auth_page()
else:
    set_background()
    profile_dropdown()

    if 'page' not in st.session_state:
        st.session_state['page'] = 'home'

    if st.session_state['page'] == 'home':
        st.title("Brain Tumor Detection")
        st.markdown("---")
        st.write('Enter your Image')

        uploaded_files = st.file_uploader("Select image(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

        if uploaded_files:
            # Load YOLO model once
            model = YOLO("best.pt")

            conn = get_db_connection()
            cursor = conn.cursor()
            user_id = st.session_state.get('user_id')

            for i, uploaded_file in enumerate(uploaded_files):
                binary_data = uploaded_file.read()

                # Reload file pointer for OpenCV/PIL
                uploaded_file.seek(0)
                pil_img = Image.open(uploaded_file).convert("RGB")
                img_cv2 = np.array(pil_img)[:, :, ::-1].copy()  # Convert PIL → OpenCV (BGR)

                # Predict with YOLO
                results = model(img_cv2)[0]

                prediction = "Negative (No Tumor)"
                confidence = 0.0

                if len(results.boxes) > 0:
                    # Get best detection (highest confidence)
                    best_box = max(results.boxes.data.tolist(), key=lambda x: x[4])
                    x1, y1, x2, y2, score, class_id = best_box
                    confidence = float(score)

                    if confidence >= 0.5:
                        prediction = "Positive (Tumor)"
                        # Draw green box
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        cv2.rectangle(img_cv2, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        cv2.putText(img_cv2, f"{confidence:.2%}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Convert back for Streamlit display
                img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)

                # Show details
                st.subheader("Details")
                st.write("Prediction:", prediction)
                st.write("Confidence:", f"{confidence:.2%}")

                # Show image
                st.image(img_rgb, caption=f"Condition: {prediction}")


                now = datetime.now()

                # Save image + results into DB
                sql = "INSERT INTO images (id, name, photo, acc, time) VALUES (%s, %s, %s, %s, %s)"
                cursor.execute(sql, (user_id, uploaded_file.name, binary_data, confidence, now))

                # PDF Report
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("Generate Report", key=f"report_{i}"):
                    pil_img_report = Image.fromarray(img_rgb)  # Convert numpy array → PIL
                    pdf_buffer = generate_pdf(user_id, prediction, confidence, pil_img_report)
                    st.download_button(
                        label='Download Report',
                        data=pdf_buffer,
                        file_name=f"report_{user_id}_{i}.pdf",
                        mime="application/pdf"
                    )

            conn.commit()
            cursor.close()
            conn.close()

        # uploaded_files = st.file_uploader("Select image(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        # if uploaded_files:
        #     model = load_model("cnn-parameters-improvement-10-0.92.keras")
        #     img_height, img_width = 240, 240
        #     conn = get_db_connection()
        #     cursor = conn.cursor()
        #     user_id = st.session_state.get('user_id')
        #
        #     for i, uploaded_file in enumerate(uploaded_files):
        #         binary_data = uploaded_file.read()
        #
        #         # Load and preprocess image
        #         img = image.load_img(uploaded_file, target_size=(img_height, img_width))
        #         img_array = np.expand_dims(image.img_to_array(img) / 255.0, axis=0)
        #
        #         # Predict
        #         pre = model.predict(img_array)
        #         predicted_class = np.argmax(pre)
        #         if pre > 0.5:
        #             predction = "Tumor Detected"
        #             pred_color = colors.red
        #         else:
        #             predction = "Tumor not Detected"
        #             pred_color = colors.green
        #         st.subheader("Details")
        #         st.write("User ID:", user_id)
        #         st.write("Prediction:", predction)
        #         st.write("Confidence:", f"{pre}%")
        #         st.write("Generated At:", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        #
        #         # Show result
        #         st.image(img, caption=f"Condition: {predction}")
        #
        #         now = datetime.now()
        #         if isinstance(pre, np.ndarray):
        #             pre = float(pre[0])
        #
        #         # Save image into DB
        #         sql = "INSERT INTO images (id, name, photo, acc, time) VALUES (%s, %s, %s, %s, %s)"
        #         cursor.execute(sql, (user_id, uploaded_file.name, binary_data, pre, now))
        #
        #         # Action on form submit
        #         st.markdown("<br>", unsafe_allow_html=True)
        #         if st.button("Generate Report", key=f"report_{i}"):
        #             pil_img = Image.open(uploaded_file)
        #             pdf_buffer = generate_pdf(user_id, predction, pre, pil_img)
        #             st.download_button(
        #                 label='download now',
        #                 data=pdf_buffer,
        #                 file_name=f"report_{user_id}_{i}.pdf",
        #                 mime="application/pdf"
        #             )
        #
        #     conn.commit()
        #     cursor.close()
        #     conn.close()
