import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

from skimage import color, measure
from skimage.filters import gaussian, threshold_otsu
from skimage.io import imread
from skimage.metrics import structural_similarity
from skimage.transform import resize
from sklearn.preprocessing import MinMaxScaler

from Ecg import ECG


st.set_page_config(page_title="ECG 4-Class Prediction", layout="wide")

APP_DIR = Path(__file__).resolve().parent
TRAIN_SIGNAL_LENGTH = 3060   # 12 leads x 255 points


# =========================
# Helper functions
# =========================
def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] == 4:
        gray = color.rgb2gray(image[:, :, :3])
    else:
        gray = color.rgb2gray(image)
    return gray


def load_uploaded_image(uploaded_file) -> np.ndarray:
    return imread(uploaded_file)


def resize_for_pipeline(image: np.ndarray) -> np.ndarray:
    return resize(
        image,
        (1572, 2213),
        preserve_range=True,
        anti_aliasing=True
    )


def divide_leads(image_resized: np.ndarray):
    lead_1 = image_resized[300:600, 150:643]
    lead_2 = image_resized[300:600, 646:1135]
    lead_3 = image_resized[300:600, 1140:1625]
    lead_4 = image_resized[300:600, 1630:2125]

    lead_5 = image_resized[600:900, 150:643]
    lead_6 = image_resized[600:900, 646:1135]
    lead_7 = image_resized[600:900, 1140:1625]
    lead_8 = image_resized[600:900, 1630:2125]

    lead_9 = image_resized[900:1200, 150:643]
    lead_10 = image_resized[900:1200, 646:1135]
    lead_11 = image_resized[900:1200, 1140:1625]
    lead_12 = image_resized[900:1200, 1630:2125]

    # long lead / rhythm strip
    lead_13 = image_resized[1250:1480, 150:2125]

    leads = [
        lead_1, lead_2, lead_3, lead_4,
        lead_5, lead_6, lead_7, lead_8,
        lead_9, lead_10, lead_11, lead_12,
        lead_13
    ]
    return leads


def preprocess_lead(lead_img: np.ndarray, sigma: float = 0.9, out_shape=(300, 450)) -> np.ndarray:
    gray = to_gray(lead_img)
    blurred = gaussian(gray, sigma=sigma)
    thresh = threshold_otsu(blurred)

    binary = (blurred < thresh).astype(np.float32)

    binary = resize(
        binary,
        out_shape,
        anti_aliasing=False,
        preserve_range=True
    )

    binary = (binary > 0.5).astype(np.float32)
    return binary


def extract_contour_signal(binary_img: np.ndarray) -> np.ndarray:
    contours = measure.find_contours(binary_img, 0.8)
    if len(contours) == 0:
        raise ValueError("Không tìm thấy contour từ lead đã preprocess.")

    contour = max(contours, key=lambda x: x.shape[0])
    contour = resize(contour, (255, 2), anti_aliasing=True)
    return contour


def contour_to_scaled_series(contour: np.ndarray) -> np.ndarray:
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(contour)
    return scaled[:, 0].astype(np.float32)


def build_1d_signal(leads):
    """
    Pipeline cũ:
    - vẫn chia đủ 13 lead
    - nhưng chỉ dùng 12 lead đầu để tạo tín hiệu đưa vào model
    """
    processed_leads = []
    contours = []
    one_d_signals = []

    # preprocess đủ 13 lead để hiển thị/debug
    for idx, lead in enumerate(leads, start=1):
        sigma = 0.7 if idx == 13 else 0.9
        binary = preprocess_lead(lead, sigma=sigma)
        contour = extract_contour_signal(binary)
        signal_1d = contour_to_scaled_series(contour)

        processed_leads.append(binary)
        contours.append(contour)
        one_d_signals.append(signal_1d)

    # CHỈ dùng 12 lead đầu để predict, khớp data train cũ
    final_signal = np.concatenate(one_d_signals[:12], axis=0)
    ecg_1dsignal = pd.DataFrame([final_signal])

    return processed_leads, contours, one_d_signals, ecg_1dsignal


def plot_divided_leads(leads):
    fig, ax = plt.subplots(4, 3, figsize=(12, 10))
    x_counter, y_counter = 0, 0

    for i, lead in enumerate(leads[:12]):
        ax[x_counter][y_counter].imshow(lead, cmap="gray")
        ax[x_counter][y_counter].axis("off")
        ax[x_counter][y_counter].set_title(f"Lead {i+1}")

        if (i + 1) % 3 == 0:
            x_counter += 1
            y_counter = 0
        else:
            y_counter += 1

    fig.tight_layout()

    fig13, ax13 = plt.subplots(figsize=(12, 3))
    ax13.imshow(leads[12], cmap="gray")
    ax13.axis("off")
    ax13.set_title("Lead 13 (display only)")

    return fig, fig13


def plot_preprocessed_leads(processed_leads):
    fig, ax = plt.subplots(4, 3, figsize=(12, 10))
    x_counter, y_counter = 0, 0

    for i, lead in enumerate(processed_leads[:12]):
        ax[x_counter][y_counter].imshow(lead, cmap="gray")
        ax[x_counter][y_counter].axis("off")
        ax[x_counter][y_counter].set_title(f"Preprocessed Lead {i+1}")

        if (i + 1) % 3 == 0:
            x_counter += 1
            y_counter = 0
        else:
            y_counter += 1

    fig.tight_layout()

    fig13, ax13 = plt.subplots(figsize=(12, 3))
    ax13.imshow(processed_leads[12], cmap="gray")
    ax13.axis("off")
    ax13.set_title("Preprocessed Lead 13 (display only)")

    return fig, fig13


def plot_contours(contours):
    fig, ax = plt.subplots(4, 3, figsize=(12, 10))
    x_counter, y_counter = 0, 0

    for i, contour in enumerate(contours[:12]):
        ax[x_counter][y_counter].plot(contour[:, 1], contour[:, 0], linewidth=1, color="black")
        ax[x_counter][y_counter].axis("image")
        ax[x_counter][y_counter].set_title(f"Contour Lead {i+1}")

        if (i + 1) % 3 == 0:
            x_counter += 1
            y_counter = 0
        else:
            y_counter += 1

    fig.tight_layout()

    fig13, ax13 = plt.subplots(figsize=(12, 3))
    ax13.plot(contours[12][:, 1], contours[12][:, 0], linewidth=1, color="black")
    ax13.axis("image")
    ax13.set_title("Contour Lead 13 (display only)")

    return fig, fig13


def maybe_check_similarity(gray_img: np.ndarray):
    """
    Check nhẹ theo app cũ.
    Nếu thiếu ảnh mẫu thì bỏ qua.
    """
    sample_names = ["PMI(1).jpg", "HB(6).jpg", "Normal(1).jpg", "MI(1).jpg"]
    sample_paths = [APP_DIR / name for name in sample_names]

    if not all(p.exists() for p in sample_paths):
        return None

    scores = []
    for p in sample_paths:
        img = imread(str(p))
        img = resize_for_pipeline(img)
        img = to_gray(img)
        scores.append(structural_similarity(gray_img, img))

    return max(scores)


# =========================
# Streamlit UI
# =========================
st.title("ECG 4-Class Prediction with ECGTransForm")
# st.subheader("Per-class Accuracy (one-vs-rest)")

per_class_acc = {
    "AHB": 0.9500,
    "MI": 0.9929,
    "NORMAL": 0.9643,
    "PM": 0.9643,
}

c1, c2, c3, c4 = st.columns(4)
c1.metric("AHB", f"{per_class_acc['AHB']:.4f}")
c2.metric("MI", f"{per_class_acc['MI']:.4f}")
c3.metric("NORMAL", f"{per_class_acc['NORMAL']:.4f}")
c4.metric("PM", f"{per_class_acc['PM']:.4f}")

# per_class_acc_df = pd.DataFrame({
#     "Class": ["AHB", "MI", "NORMAL", "PM"],
#     "Per-class Accuracy": [0.9500, 0.9929, 0.9643, 0.9643],
# })

# st.dataframe(per_class_acc_df, use_container_width=True)
# st.caption("Predict pipeline follows old training logic: image has 13 leads, model uses first 12 leads for signal vector.")

uploaded_file = st.file_uploader(
    "Choose an ECG image",
    type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"]
)

if uploaded_file is not None:
    try:
        ecg = ECG()

        image = load_uploaded_image(uploaded_file)
        image_resized = resize_for_pipeline(image)
        image_gray = to_gray(image_resized)

        similarity_score = maybe_check_similarity(image_gray)

        if similarity_score is not None:
            st.write(f"Similarity score: {similarity_score:.4f}")
            if similarity_score <= 0.70:
                st.warning("Ảnh này khá khác format mẫu. App vẫn sẽ thử xử lý, nhưng kết quả có thể không ổn định.")

        st.subheader("Uploaded ECG Image")
        st.image(image, use_container_width=True)

        with st.expander("Gray Scale Image", expanded=False):
            st.image(image_gray, clamp=True, use_container_width=True)

        leads = divide_leads(image_resized)

        fig_leads, fig_lead13 = plot_divided_leads(leads)
        with st.expander("Dividing Leads", expanded=False):
            st.pyplot(fig_leads)
            st.pyplot(fig_lead13)

        processed_leads, contours, one_d_signals, ecg_1dsignal = build_1d_signal(leads)

        fig_pre, fig_pre13 = plot_preprocessed_leads(processed_leads)
        with st.expander("Preprocessed Leads", expanded=False):
            st.pyplot(fig_pre)
            st.pyplot(fig_pre13)

        fig_contour, fig_contour13 = plot_contours(contours)
        with st.expander("Contour Leads", expanded=False):
            st.pyplot(fig_contour)
            st.pyplot(fig_contour13)

        web_len = ecg_1dsignal.shape[1]

        with st.expander("1D Signal Preview", expanded=False):
            st.write("Shape:", ecg_1dsignal.shape)
            st.write("Train signal length:", TRAIN_SIGNAL_LENGTH)
            st.write("Web signal length:", web_len)
            st.dataframe(ecg_1dsignal.iloc[:, :50])

        if web_len != TRAIN_SIGNAL_LENGTH:
            st.error(f"Length mismatch! Train={TRAIN_SIGNAL_LENGTH}, Web={web_len}")
        else:
            st.success(f"Signal length matches training data: {web_len}")

        prediction = ecg.ModelLoad_predict_transformer_4class(ecg_1dsignal)

        st.subheader("Prediction")
        st.success(prediction)

    except Exception as e:
        st.error(f"Đã xảy ra lỗi khi xử lý ảnh hoặc dự đoán: {e}")