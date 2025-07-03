import streamlit as st
import google.generativeai as genai

from utils.document_loaded import load_pdf,load_docx,load_txt,chunk_text
from utils.text_embedder import embed_chunks,retrieve_relevant_chunks
import time

#page config
st.set_page_config(page_title="Talk with your Doc",layout='wide')
st.title('Talk to your own Doc in your own style')

#sidebar for Gemini API key
st.sidebar.header("Gemini API Setup")
user_api_key=st.sidebar.text_input('Enter your Gemini API', type='password')

if not user_api_key:
    st.warning('Please enter your Gemini key in sidebar to start')
    st.stop()

# try:
#     genai.configure(api_key=user_api_key)
#     chat_model=genai.GenerativeModel('gemini-2.0-flash-lite')
#     st.sidebar.success('Gemini Api configured')
# except Exception as e:
#     st.sidebar.error(f'API setup failed{e}')
#     st.stop()


# try:
#     genai.configure(api_key=user_api_key)
    
#     # Test the API key with a lightweight request
#     test_model = genai.GenerativeModel('gemini-2.0-flash-lite')
#     test_model.generate_content("Test connection", request_options={"timeout": 5})  # Force network call
    
#     chat_model = test_model  # Use validated model
#     st.sidebar.success('Gemini API configured')
# except Exception as e:
#     st.sidebar.error(f'API setup failed: {e}')
#     st.stop()  # Stop app immediately

try:
    genai.configure(api_key=user_api_key)
    test_model = genai.GenerativeModel('gemini-2.0-flash-lite')
    test_model.generate_content("Connection test", request_options={"timeout": 5})
    chat_model = test_model
    st.sidebar.success('Gemini API configured successfully!')
except Exception as e:
    error_msg = str(e)
    
    # Handle invalid API key case
    if "API_KEY_INVALID" in error_msg or "API key not valid" in error_msg:
        st.sidebar.error('❌ Invalid API Key. Please check your Google Gemini API key and try again.')
    
    # Handle other common errors
    elif "PERMISSION_DENIED" in error_msg:
        st.sidebar.error('🔐 Permission denied. Ensure your API key has proper access rights.')
    elif "503" in error_msg:
        st.sidebar.error('🌐 Service unavailable. Google API services might be down.')
    elif "timed out" in error_msg:
        st.sidebar.error('⌛ Connection timed out. Check your internet connection.')
    
    #Fallback for unknown errors
    else:
        st.sidebar.error(f'⚠️ API setup failed: {error_msg.split(".")[0]}')
    
    st.stop()


#uplaod doc
uploaded_file=st.file_uploader("Upload your doc (PDF,DOCX,TXT)",type=['pdf','docx','txt'])

#check if new file is uploaded
if uploaded_file:
    #if doc is uploaded and clear
    if 'last_uploaded_file' not in st.session_state or st.session_state['last_uploaded_file']!=uploaded_file.name:
        #New doc -reset everything
        st.session_state.clear()
        st.session_state['last_uploaded_file']=uploaded_file.name

        #reconfigure gemini
        genai.configure(api_key=user_api_key)
        chat_model=genai.GenerativeModel('gemini-2.0-flash-lite')

        file_ext=uploaded_file.name.split('.')[-1].lower()
        if file_ext=='pdf':
            raw_text=load_pdf(uploaded_file)

        elif file_ext=='docx':
            raw_text=load_docx(uploaded_file)
        elif file_ext=='txt':
            raw_text=load_txt(uploaded_file)

        else:
            st.error('Unsupported file type')
            st.stop()

        st.info('Chunking and embedding doc..')
        chunks=chunk_text(raw_text)
        embeddings=embed_chunks(chunks)

        st.session_state['chunks']=chunks
        st.session_state['embedding']=embeddings
        st.session_state['document_processed']=True

        st.success(f'Doc processed and embedded into {len(chunks)} chunks.')
    else:
        st.session_state['document_processed']=True
else:
    st.session_state['document_processed']=False

#Ask questions only after doc is processed
if st.session_state.get('document_processed',False):
    st.subheader("Ask a questions about Document")

    #ensure fresh key to clear old query when doc is changed
    query_key=f'query_input_{st.session_state['last_uploaded_file']}'
    query=st.text_input('Enter you questions:',key=query_key)

    if query:
        st.info('Retriving relevent chunks..')
        top_chunks=retrieve_relevant_chunks(
            query,
            st.session_state['chunks'],
            st.session_state['embedding'],
            top_k=7

        )
        context='\n\n'.join(top_chunks)
        prompt=f'''Answer the question based on the following conext:\n\n{context}\n\nQeestion: {query}'''

        st.info('Generating answer with gemini')
        st.markdown('### Answer')
        response_area=st.empty()
        try:
            response_stream=chat_model.generate_content(prompt,stream=True)
            full_response=''
            for chunk in response_stream:
                if chunk.text:
                    full_response+=chunk.text
                    response_area.markdown(full_response)
                    time.sleep(0.05)
        except Exception as e:
            st.error(f'Error generating response: {e}')

else:
    st.info('Please upload and process a doc to start asking questions')






       
