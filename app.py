import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# 페이지 설정
st.set_page_config(page_title="AI 인허가 자동검색", page_icon="💡", layout="centered")
st.title("💡 AI 인허가 자동검색 시스템")
st.write("업종명을 입력하면 관련 인허가 정보를 자동으로 찾아드립니다. (엑셀 데이터 기반)")

# 엑셀 데이터 불러오기
@st.cache_data
def load_permit_data():
    df = pd.read_excel("permit_data.xlsx")
    return df

df = load_permit_data()

# AI 임베딩 모델 불러오기
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# 인허가 검색 함수
def search_permits(user_input):
    # 업종명 벡터화
    query_embedding = model.encode(user_input, convert_to_tensor=True)
    # 각 행의 유사도 계산
    df["유사도"] = df["업종명"].apply(lambda x: util.cos_sim(query_embedding, model.encode(x, convert_to_tensor=True)).item())
    # 유사도 순으로 정렬
    results = df.sort_values(by="유사도", ascending=False).head(5)
    return results

# 입력창
user_input = st.text_input("🔍 업종명을 입력하세요 (예: 카페, 학원, 쇼핑몰 등)")

if st.button("검색"):
    if user_input:
        results = search_permits(user_input)
        st.subheader(f"“{user_input}” 관련 인허가 정보")
        for _, row in results.iterrows():
            with st.expander(f"📋 {row['업종명']} (유사도: {row['유사도']:.2f})"):
                st.write(f"**관련 부서:** {row['관련부서']}")
                st.write(f"**근거 법령:** {row['근거법령']}")
                st.write(f"**필요 서류:** {row['필요서류']}")
    else:
        st.warning("업종명을 입력해주세요!")

