import streamlit as st
from inference import main as st_main

if __name__ == '__main__':
    st.set_page_config(page_title="Image Classification")

    st_main()
