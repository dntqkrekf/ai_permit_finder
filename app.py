import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util

st.set_page_config(page_title="AI 인허가 자동검색", page_icon="💡", layout="centered")
st.title("💡 AI 인허가 자동검색 시스템")
st.write("업종명을 입력하면 관련 인허가 정보를 자동으로 찾아드립니다. (엑셀 데이터 기반)")

# --- 엑셀 파일 불러오기 ---
@st.cache_data
def load_permit_data():
    try:
        df = pd.read_excel("permit_data.xlsx")
        return df
    except FileNotFoundError:
        st.error("⚠️ permit_data.xlsx 파일이 GitHub 저장소에 없습니다.")
        return pd.DataFrame()

df = load_permit_data()

# --- AI 모델 로드 ---
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# --- 검색 기능 ---
def find_best_match(user_input):
    if df.empty:
        return None
    permits = df["업종명"].tolist()
    embeddings = model.encode(permits, convert_to_tensor=True)
    query_embedding = model.encode(user_input, convert_to_tensor=True)
    cos_scores = util.cos_sim(query_embedding, embeddings)[0]
    best_idx = int(cos_scores.argmax())  # 가장 유사한 항목의 인덱스
    best_score = float(cos_scores[best_idx])
    result = df.iloc[best_idx]
    return result, best_score

# --- 사용자 입력 ---
user_input = st.text_input("🔍 업종명을 입력하세요 (예: 음식, 카페, 학원 등)")
if st.button("검색"):
    if not user_input.strip():
        st.warning("업종명을 입력하세요.")
    else:
        match, score = find_best_match(user_input)
        if match is not None:
            st.markdown(f"### 🔎 “{user_input}”와 가장 유사한 인허가 정보 (유사도: {score:.2f})")
            with st.expander(f"📌 {match['업종명']}"):
                st.write(f"**관련 부서:** {match['관련부서']}")
                st.write(f"**근거 법령:** {match['근거법령']}")
                st.write(f"**필요 서류:** {match['필요서류']}")
        else:
            st.error("검색 결과가 없습니다.")


