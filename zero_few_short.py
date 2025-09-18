import streamlit as st
import ollama
from pypdf import PdfReader

# Function to extract text from PDF
def extract_pdf_text(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text.strip()

# Zero-Shot Summarizer
def summarize_zero_shot(text, length="short"):
    prompt = f"Summarize the following text in {length} form (3-4 bullet points):\n\n{text}"
    
    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}]
    )
    return response['message']['content']

# Few-Shot Summarizer
def summarize_few_shot(text, length="short"):
    prompt = f"""
You are a summarization assistant. Summarize text into 3 concise bullet points.

Example 1:
Text: "Python is a popular programming language. It is used in web development, data science, and AI. Its simplicity makes it beginner-friendly."
Summary:
- Python is widely used in web development, data science, and AI.
- Known for its simplicity and beginner-friendly syntax.
- Popular programming language with diverse applications.

Example 2:
Text: "Electric cars are becoming more popular. They help reduce carbon emissions. However, charging infrastructure is still limited in many countries."
Summary:
- Electric cars are growing in popularity.
- They reduce carbon emissions.
- Charging stations are still limited worldwide.

Now summarize this text in {length} form:
{text}
Summary:
"""
    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}]
    )
    return response['message']['content']

# Streamlit UI
st.title("📄 Zero-Shot & Few-Shot Summarizer (Ollama)")

# Custom Background with CSS
page_bg_css = """
<style>
/* Background color */
.stApp {
    background-color: #f0f8ff; /* light blue */
}

/* Optional: Add background image */
.stApp {
    background: url("https://images.unsplash.com/photo-1507525428034-b723cf961d3e") no-repeat center center fixed;
    background-size: cover;
}
</style>
"""

st.markdown(page_bg_css, unsafe_allow_html=True)


# Choose mode
mode = st.radio("Choose mode:", ["Zero-Shot", "Few-Shot"])

# Choose input type
option = st.radio("Choose input type:", ["Text", "PDF"])

user_text = ""

if option == "Text":
    user_text = st.text_area("Paste your text here:", height=200)

elif option == "PDF":
    pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    if pdf_file is not None:
        with st.spinner("Extracting text from PDF..."):
            user_text = extract_pdf_text(pdf_file)
        st.success("✅ PDF text extracted!")

# Choose summary length
length_option = st.selectbox("Choose summary length:", ["short", "medium", "detailed"])

# Summarize button
if st.button("Summarize"):
    if user_text.strip():
        with st.spinner("Generating summary..."):
            if mode == "Zero-Shot":
                summary = summarize_zero_shot(user_text, length_option)
            else:
                summary = summarize_few_shot(user_text, length_option)
        st.subheader("🔹 Summary:")
        st.write(summary)
    else:
        st.warning("⚠️ Please enter text or upload a PDF to summarize.")
