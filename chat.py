import openai
from streamlit_chat import message
from langchain_tools import ag_plugin
import json
import requests
from PIL import Image
import streamlit as st
from io import BytesIO

def search(prompt):
    st.session_state['messages'].append({"role": "user", "content": prompt})
    completion = json.loads(ag_plugin(prompt))

    print(type(completion))
    print(completion)
    
    response = completion["choices"][0]["message"]["content"]
    
    st.session_state['messages'].append({"role": "assistant", "content": response})

    total_tokens = completion["usage"]["total_tokens"]
    prompt_tokens = completion["usage"]["prompt_tokens"]
    completion_tokens = completion["usage"]["completion_tokens"]

    return response, total_tokens, prompt_tokens, completion_tokens


# Setting page title and header
st.set_page_config(page_title="ChatBot", page_icon=":robot_face:", layout="wide")
st.markdown("<h1 style='text-align: center;'>Langchain Chatbot</h1>", unsafe_allow_html=True)

# Initialise session state variables
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
if 'messages' not in st.session_state:
    st.session_state['messages'] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]

# container for chat history
with st.expander("Chatbot", expanded=True):
    response_container = st.container()
    with response_container:
        st.write("##")
# container for text box
container = st.container()

with container:
    with st.form(key='user_send', clear_on_submit=True):
        user_input = st.text_area("You:", key='input')
        submit_button = st.form_submit_button(label='Send') 

        if submit_button and user_input:
            output, total_tokens, prompt_tokens, completion_tokens = search(user_input)

            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)


if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
            if "image" in st.session_state["generated"][i]:
                address = st.session_state["generated"][i].split(":")[-1]
                response = requests.get("https:" + address)

                img = Image.open(BytesIO(response.content))
                message(f"Here is the image: https:{address}", key=str(i))
                image_em1, image_em2 = st.columns([0.5,12])
                with image_em2:
                    st.image(img)
            else:
                message(st.session_state["generated"][i], key=str(i))

