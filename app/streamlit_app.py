import streamlit as st
from qa_chain import get_qa_chain

# Cache the chain so it's not recreated on every question
@st.cache_resource
def load_chain():
    return get_qa_chain()

qa_chain = load_chain()

st.set_page_config("💳 CrediBot: Your Credit Card Policy Assistant", layout="wide")
st.title("💳 CrediBot: Your Credit Card Policy Assistant")
st.markdown("Ask me anything about **Chase** or **AMEX** credit card policies ")

user_question = st.text_input("Ask a question:")

if user_question:
    with st.spinner("Searching documents..."):
        qa_chain = get_qa_chain()
        result = qa_chain({"query": user_question})

    # Extract and display answer
    answer = result["result"]
    st.subheader("Answer:")
    st.write(answer)


