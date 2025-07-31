
import streamlit as st
from utils.api_calls import *

st.set_page_config(layout="wide")
st.title("Model (Transformer, GAN, DDIM, RAG Bot)")

tab1, tab2, tab3, tab4 = st.tabs([
    "1️⃣ Transformer Sentiment Analysis",
    "2️⃣ DCGAN Image Generator",
    "3️⃣ DDIM Text-to-Image",
    "4️⃣ RAG Chatbot"
])

with tab1:
    st.subheader("Sentiment Analysis (Transformer)")
    text = st.text_area("Enter text:")
    if st.button("Analyze Sentiment"):
        result = call_transformer_model(text)
        if result:
            st.success(f"Prediction: {result['prediction']}")
            interpretation = result.get("interpretation", {})
            if interpretation:
                st.markdown("**Interpretation Weights:**")
                for word, weight in interpretation.items():
                    st.markdown(f"`{word}`: {weight:.5f}")
        else:
            st.error("Prediction failed or unauthorized")

with tab2:
    st.subheader("DCGAN Image Generator")
    if st.button("Generate GAN Image"):
        img_url = call_dcgan_model()
        if img_url:
            st.image(img_url, caption="Generated Image")
        else:
            st.error("GAN generation failed or unauthorized")

with tab3:
    st.subheader("DDIM Text-to-Image")
    caption = st.text_input("Enter caption for DDIM:")
    if st.button("Generate DDIM Image"):
        img_url = call_ddim_model(caption)
        if img_url:
            st.image(img_url, caption="Generated from DDIM")
        else:
            st.error("DDIM generation failed")

with tab4:
    st.subheader("RAG-based Chatbot")
    query = st.text_input("Enter your question:")
    if st.button("Ask Bot"):
        answer = call_rag_model(query)
        if answer:
            st.success(answer)
        else:
            st.error("RAG bot failed")
