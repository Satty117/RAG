import os
import time
from dotenv import load_dotenv
import streamlit as st

from langchain_core.prompts import ChatPromptTemplate
import web as ws

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
st.title("LLM-based Web Content Reader")


def hide_streamlit_style():
    """Function to hide Streamlit's default UI elements for a cleaner look."""
    st.markdown(
        """
        <style>
        #MainMenu
        footer {visibility: hidden;}
        header {visibility: hidden;}
        </style>
        """,
        unsafe_allow_html=True,
    )


hide_streamlit_style()


user_query = st.text_input("Enter your question here:")

if user_query:

    st.info("Fetching web pages...")
    links = ws.fetch_web_page(user_query)

    if not links:
        st.warning("No results found for the query. Please try again with a different query.")
    else:
        st.info("Scraping content from the fetched links...")
        top_results = ws.page_scrape(links, user_query)


        st.info("Generating summary...")
        summary = ws.llm_summariser(top_results, user_query)

        start = time.process_time()
        prompt_template = ChatPromptTemplate.from_template(
            f"""
            Answer the question based on the provided context only.
            Please provide the most accurate response based on the question.
            <context>
            {summary}
            <context>
            Question: {user_query}
            """
        )
        end_time = time.process_time() - start




        st.write(f"Response time: {end_time:.2f} seconds")
        st.subheader("AI Response:")
        st.write(summary)


        with st.expander("Document Similarity Search Results"):

            if summary:
                st.write(summary)
                st.write("-----------------------------------")
            else:
                st.info("No context available for similarity search results.")
else:
    st.warning("Please enter a query to start.")
