import streamlit as st
from sentence_transformers import SentenceTransformer, util

st.set_page_config(page_title="AI 인허가 자동검색", page_icon="💡", layout="centered")
st.title("💡 AI 인허가 자동검색 시스템")
st.write("사업 내용을 입력하면 AI가 관련 인허가를 자동으로 찾아드립니다.")

permits = [
    {
        "name": "통신판매업 신고",
        "keywords": "온라인 쇼핑몰, 인터넷 판매, 전자상거래, 의류 판매",
        "agency": "관할 구청 지역경제과",
        "law": "전자상거래법 제12조",
        "required_docs": ["신고서", "사업자등록증 사본", "도메인 등록 증명서"]
    },
    {
        "name": "식품위생 영업신고",
        "keywords": "음식점, 카페, 주점, 제과점, 식품 제조업",
        "agency": "구청 위생과",
        "law": "식품위생법 제37조",
        "required_docs": ["영업신고서", "위생교육이수증", "임대차계약서"]
    },
    {
        "name": "옥외광고물 표시허가",
        "keywords": "간판, 현수막, 옥외광고, 전광판",
        "agency": "구청 도시디자인과",
        "law": "옥외광고물 등 관리법 제3조",
        "required_docs": ["허가신청서", "설치도면", "건물주 동의서"]
    }
]

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')  # 가벼운 모델로 교체

model = load_model()

def find_related_permits(user_input):
    query_emb = model.encode(user_input)
    results = []
    for permit in permits:
        score = util.cos_sim(query_emb, model.encode(permit["keywords"])).item()
        results.append((permit, score))
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:3]

user_input = st.text_input("👉 사업 내용을 입력하세요", placeholder="예: 온라인으로 의류를 판매하고 싶어요")

if st.button("🔍 인허가 검색"):
    if user_input.strip():
        with st.spinner("AI가 인허가를 분석 중입니다..."):
            results = find_related_permits(user_input)
        st.subheader(f"🔍 “{user_input}”에 대한 추천 결과")
        for permit, score in results:
            with st.expander(f"🟢 {permit['name']} (유사도 {score:.2f})", expanded=True):
                st.write(f"**근거법령:** {permit['law']}")
                st.write(f"**담당기관:** {permit['agency']}")
                st.write(f"**필요서류:** {', '.join(permit['required_docs'])}")
    else:
        st.warning("사업 내용을 입력해주세요.")
