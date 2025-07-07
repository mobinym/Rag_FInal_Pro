# analyzer_app.py

import streamlit as st
import pandas as pd
import os

# --- تنظیمات اولیه صفحه ---
st.set_page_config(page_title="داشبورد تحلیل فیدبک RAG", layout="wide")

# --- مسیر فایل CSV ---
FEEDBACK_FILE_PATH = "feedback_logs/feedback.csv"

# --- تابع برای بارگذاری داده‌ها (با قابلیت کش شدن) ---
@st.cache_data
def load_data(file_path):
    """داده‌ها را از فایل CSV می‌خواند و به صورت یک DataFrame برمی‌گرداند."""
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    return None

# --- عنوان اصلی داشبورد ---
st.title("📊 داشبورد تحلیل فیدبک سیستم RAG")
st.markdown("در این صفحه می‌توانید عملکرد سیستم را بر اساس فیدبک‌های ثبت شده، تحلیل و بررسی کنید.")

# --- بارگذاری داده‌ها ---
df = load_data(FEEDBACK_FILE_PATH)

if df is None or df.empty:
    st.warning("فایل فیدبک یافت نشد یا خالی است. لطفاً ابتدا چند سوال از سیستم اصلی بپرسید و فیدبک ثبت کنید.")
else:
    # --- بخش ۱: نمایش آمار کلی ---
    st.header("📈 آمار کلی در یک نگاه")

    # محاسبه آمار
    total_feedbacks = len(df)
    good_feedbacks = len(df[df['feedback_status'] == 'good'])
    bad_feedbacks = total_feedbacks - good_feedbacks
    success_rate = (good_feedbacks / total_feedbacks) * 100 if total_feedbacks > 0 else 0
    avg_retrieval_time = df['retrieval_time_sec'].mean()
    avg_generation_time = df['generation_time_sec'].mean()

    # نمایش در ستون‌های مجزا
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("تعداد کل فیدبک‌ها", f"{total_feedbacks} عدد")
    col2.metric("درصد پاسخ‌های موفق", f"{success_rate:.1f}%")
    col3.metric("میانگین زمان بازیابی", f"{avg_retrieval_time:.2f} ثانیه")
    col4.metric("میانگین زمان تولید پاسخ", f"{avg_generation_time:.2f} ثانیه")
    
    st.divider()

    # --- بخش ۲: فیلتر کردن و نمایش داده‌ها ---
    st.header("📋 جدول کامل فیدبک‌ها")

    # فیلتر بر اساس وضعیت فیدبک
    all_statuses = df['feedback_status'].unique()
    selected_statuses = st.multiselect(
        "فیلتر بر اساس وضعیت فیدبک:",
        options=all_statuses,
        default=all_statuses
    )

    # اعمال فیلتر
    filtered_df = df[df['feedback_status'].isin(selected_statuses)]

    # نمایش جدول تعاملی (می‌توانید ستون‌ها را جابجا و مرتب کنید)
    st.dataframe(filtered_df)
    st.info("💡 برای مشاهده کامل متن هر ستون، می‌توانید لبه آن را با ماوس بکشید و اندازه آن را تغییر دهید.")
    
    st.divider()

    # --- بخش ۳: بررسی دقیق یک ردیف خاص ---
    st.header("🔍 بررسی جزئیات یک ردیف")
    
    # انتخاب ردیف بر اساس ایندکس
    selected_index = st.number_input(
        "ایندکس (شماره ردیف) مورد نظر برای بررسی را وارد کنید:",
        min_value=0,
        max_value=len(df)-1,
        step=1
    )

    if selected_index in df.index:
        selected_row = df.loc[selected_index]

        st.markdown(f"**سوال:** {selected_row['question']}")
        st.markdown(f"**پاسخ خام سیستم:** {selected_row['raw_answer']}")
        if pd.notna(selected_row['corrected_answer']):
            st.markdown(f"**پاسخ اصلاح شده:** {selected_row['corrected_answer']}")
        
        # نمایش زمینه به صورت کاملاً خوانا و مرتب
        with st.expander("نمایش زمینه (Context) ارسال شده به LLM"):
            context_parts = selected_row['retrieved_context'].split("\n\n---\n\n")
            scores = selected_row['retrieval_scores'].split(', ')
            for i, part in enumerate(context_parts):
                with st.container(border=True):
                    st.caption(f"قطعه {i+1} | امتیاز: {scores[i] if i < len(scores) else 'N/A'}")
                    st.text(part)