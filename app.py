import numpy as np
import pandas as pd
import re
import string
import spacy
import streamlit as st
import spacy_streamlit
import spacy
from textblob import TextBlob
from gensim.summarization import summarize

nlp = spacy.load("en_core_web_sm")

st.set_page_config(page_title="Text Summarization",layout="wide",initial_sidebar_state="expanded")
st.markdown(""" ### Natural Language Processing Web Application """)
raw_text = st.text_area("Input text here", height=350)
doc = nlp(raw_text)

with st.sidebar:
    st.header("Navigation")
    nav_list = ["Sentence Segmentation", "Tokenization", "POS Tagging", "Lemmatization", "Name Entity Recognition",
               "Summarization", "Sentiment Analysis"]
    choice = st.radio(label="Go to", options=nav_list, index=0)
    with st.expander(label="Developed By", expanded=False):
        st.markdown("""
            #### Bhaswati Roy 
            #### Arya Gupta
            """)
        # st.balloons()
    st.text(" ")
    st.text(" ")
    st.markdown(""" **Source Code ** [Github](https://github.com/BhaswatiRoy/Language-Processor)
               """)

if raw_text is not None:
    if choice == "Sentence Segmentation":
            l = []
            if st.button("Segmentize"):
                st.write(f""" There are **{len(list(doc.sents))} Sentences** in this text dataset.""")
                for sent in doc.sents:
                     l.append(sent)
                d={"Sentences":l}
                df=pd.DataFrame(data=d)
                st.write(df)

    if choice == "Tokenization":
        if st.button("Tokenize"):
            spacy_streamlit.visualize_tokens(doc=doc, attrs=["text"])

    if choice == "POS Tagging":
        if st.button("POS Analysis"):
            spacy_streamlit.visualize_tokens(doc=doc, attrs=["text","pos_"])

    if choice == "Lemmatization":
        if st.button("Lemmatize"):
            spacy_streamlit.visualize_tokens(doc=doc, attrs=["text", "pos_", "lemma_", "shape_"])

    if choice == "Name Entity Recognition":
        if st.button("Analyze"):
            spacy_streamlit.visualize_ner(doc=doc, labels=nlp.get_pipe("ner").labels)

    if choice == "Summarization":
        sum_words_count = st.slider(label="Words in Summary", min_value=50, max_value=500, step=25, value=100)
        if st.button("Summarize"):
            req_text = summarize(text=raw_text, word_count=sum_words_count, )
            st.write(req_text)
    
    if choice == "Sentiment Analysis":
        if st.button("Analyze Sentiments"):
            sent=raw_text
            # sent2=st.text_input(label="Enter Sentences")
            blob = TextBlob(raw_text)
            polarity, subjectivity = blob.sentiment
            # if st.button("Predict"):
            if polarity > 0:
                st.success("Positive Text!!")
            elif polarity < 0:
                st.error("Negetive Text!!")
            else:
                st.warning("Neutral Text!!")

else:
    pass
