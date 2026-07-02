import streamlit as st
from PIL import Image
import numpy as np
import sys
import os
import math

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.inference import Predictor
from core.config import Config
from core.preprocessing import apply_clahe

# ====================== تنظیمات صفحه و CSS  ======================
st.set_page_config(
    page_title="نسخه دمو تشخیص سرطان پستان با هوش مصنوعی",
    page_icon="🎗️",
    layout="centered"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] {font-family: 'Vazirmatn', sans-serif; direction: rtl; text-align: right;}
.message-card {border-radius: 16px; padding: 16px; margin: 12px 0; box-shadow: 0 4px 12px rgba(0,0,0,0.08); max-width: 700px; background-color: #f7f7f7;}
.user-msg {background-color: #6068d6; color: white; text-align: left;}
.bot-msg {background-color: #ffffff; color: #222; text-align: right;}
div[data-testid="stButton"] > button {background-color: #6068d6 !important; color: white !important; border-radius: 20px !important; height: 3rem; font-size: 1rem !important; font-weight: 600; width: auto !important; padding: 0 20px; border: none !important;}
.stFileUpload {border: 1px solid #ccc; border-radius: 20px; padding: 8px 12px; max-width: 700px; margin: auto; background-color: #fafafa;}
</style>
""", unsafe_allow_html=True)

# ====================== عنوان  ======================
st.markdown("<h1 style='text-align: center; color: #222;'>تشخیص سرطان پستان با هوش مصنوعی</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #555; font-size: 1.1rem;'>آپلود تصویر ماموگرافی و دریافت تحلیل خودکار</p>", unsafe_allow_html=True)
st.markdown("---")

# ====================== بارگذاری مدل ======================
@st.cache_resource
def load_predictor():
    return Predictor()

predictor = load_predictor()

# نمایش وضعیت مدل
if predictor.mock_mode:
    st.sidebar.warning("⚠️ حالت آزمایشی (Mock Mode) - مدل واقعی بارگذاری نشده")
else:
    st.sidebar.success("✅ مدل با موفقیت بارگذاری شد")

# ====================== آپلودر ======================
st.markdown("### 📤 آپلود تصویر ماموگرافی")
uploaded_file = st.file_uploader(
    "", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"], label_visibility="collapsed"
)
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.95rem; margin-top: -10px; margin-bottom: 20px;'>
    فرمت‌های مجاز: PNG, JPG, BMP, TIFF &nbsp;&nbsp;|&nbsp;&nbsp; حداکثر حجم: ۱۰ مگابایت
</div>
""", unsafe_allow_html=True)

MAX_SIZE = 10 * 1024 * 1024

# ====================== توابع نمایش پیام ======================
def show_user_message(text):
    st.markdown(f"<div class='message-card user-msg'>{text}</div>", unsafe_allow_html=True)

def show_bot_message(text):
    st.markdown(f"<div class='message-card bot-msg'>{text}</div>", unsafe_allow_html=True)

# ====================== پردازش تصویر ======================
if uploaded_file is not None:
    if uploaded_file.size > MAX_SIZE:
        show_bot_message(f"⚠️ حجم فایل بیش از ۱۰ مگابایت است!")
        st.stop()

    image = Image.open(uploaded_file)
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")

    show_user_message(f"فایل آپلود شد: {uploaded_file.name}")

    # نمایش تصویر اصلی و پیش‌پردازش شده
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="تصویر اصلی", use_container_width=True)
    
    with col2:
        # نمایش تصویر CLAHE
        image_np = np.array(image.convert("RGB"))
        clahe_image = apply_clahe(image_np)
        st.image(clahe_image, caption="تصویر پس از CLAHE", use_container_width=True)

    if st.button("🚀 شروع تحلیل"):
        show_bot_message("⏳ در حال پردازش تصویر و اجرای مدل هوش مصنوعی...")
        
        # پیش‌بینی با استفاده از Predictor
        result = predictor.predict(image)
        
        # Yellow Flag Warning
        if result['yellow_flag']:
            st.warning("⚠️ **هشدار Yellow Flag**: نتیجه تشخیص مدل نزدیک به مرز بوده و با عدم اطمینان همراه است. پیشنهاد می‌شود تصویر توسط پزشک متخصص بررسی شود.")
        
        # نمایش نتایج
        is_malignant = result["class"] == "Malignant"
        result_text = "سرطانی (Malignant)" if is_malignant else "خوش‌خیم (Benign)"
        emoji = "🔴" if is_malignant else "🟢"

        show_bot_message(f"{emoji} **نتیجه پیش‌بینی:** {result_text}<br>📊 **درجه اطمینان مدل:** {result['confidence']:.1%}")
        
        # نمایش احتمالات هر کلاس
        st.subheader("📊 احتمالات کلاس‌ها")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Benign (خوش‌خیم)", f"{result['raw_probabilities'][0]:.1%}")
        with col2:
            st.metric("Malignant (سرطانی)", f"{result['raw_probabilities'][1]:.1%}")

        # Mock Mode Indicator
        if result['mock']:
            st.info("ℹ️ این نتیجه توسط Mock Backend تولید شده است (مدل واقعی بارگذاری نشده)")

        # جشن برای نتیجه خوب
        if not is_malignant:
            st.balloons()

        # Grad-CAM Placeholder
        with st.expander("🔍 نقشه توجه مدل (Grad-CAM)"):
            st.info("قابلیت Grad-CAM در نسخه‌های بعدی اضافه خواهد شد.")
            st.write("این بخش نقشه حرارتی مناطق مهم تصویر را نشان می‌دهد که مدل برای تصمیم‌گیری به آن‌ها توجه کرده است.")

        # خروجی خام
        with st.expander("🔍 مشاهده خروجی خام مدل (JSON)"):
            st.json(result)

else:
    show_bot_message("⚠️ هنوز فایلی انتخاب نشده است.")
    st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcREodlGWaMuDRtmStpOT0djJ5BP3a4_nHlqTg&s",
             use_container_width=True, caption="تصویر نمونه از ماموگرافی دیجیتال")

# ====================== Footer ======================
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #888; font-size: 0.9rem; margin-bottom: 2rem;'>"
    "دموی آموزشی تشخیص سرطان پستان • فقط برای اهداف تحقیقاتی و آموزشی • دقت واقعی مدل ممکن است متفاوت باشد"
    "</p>",
    unsafe_allow_html=True
)
