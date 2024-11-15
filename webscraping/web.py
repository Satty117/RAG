import os
import requests
from bs4 import BeautifulSoup
import chromadb
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain.docstore.document import Document
from langchain_mistralai import MistralAIEmbeddings



load_dotenv()
client = chromadb.Client()
embeddings = MistralAIEmbeddings(model="mistral-embed",)
vector_store = Chroma(
    collection_name="embedding_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
)

groq_api_key = os.getenv("GROQ_API_KEY")

llm_new = ChatGroq(mistral_api_key = groq_api_key, model_name = "llama3-8b-8192")

def fetch_web_page(query):
    params = {
        "q": query,
        "hl": "en",
        "gl": "in",
        "start": 0,
        "num": 10
    }
    headers = {
        'user-agent': "Mozilla/5.0 (Macintosh; Intel Mac OS X 11_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.2 Safari/605.1.15"
    }
    html = requests.get("https://www.google.com/search", params=params, headers=headers, timeout=30).text
    soup = BeautifulSoup(html, 'lxml')
    ls = []
    for link in soup.select(".tF2Cxc"):
        result = link.select_one(".yuRUbf a")["href"]
        ls.append(result)
    return ls

def page_scrape(links,query):

    all_documents = []
    all_ids = []

    for link in links:
        print(link)
        headers = {
            'user-agent': "Mozilla/5.0 (Macintosh; Intel Mac OS X 11_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.2 Safari/605.1.15"
        }
        html = requests.get(link, headers=headers).text
        soup = BeautifulSoup(html, 'lxml')
        res = [s.get_text().strip() for s in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6","p"])]
        v="\n".join(res)


        document = Document(
            page_content=v,
            metadata={"source": link},
            id=len(all_documents) + 1
        )

        all_documents.append(document)
        all_ids.append(document.id)


    vector_store.add_documents(
    ids=all_ids,
    documents=all_documents,

)
    results = vector_store.similarity_search(
        query,
        k=1
    )
    print('results', results)
    top_results =  results.page_content
    return top_results


def llm_summariser(top_results,query):
    print('summarising the data')

    prompt = f"Given the following content from various sources, please summarize the content and provide most accurate answer to the following data:\n\n{top_results}\n\nQuery: {query}\nAnswer:"
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant who should summarize the content."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]


    chat_completion = llm_new.invoke(messages)
    print(chat_completion.type)

    summary = chat_completion.content
    print(summary)
    print('result is printed')
    return summary if summary else "No response Generated"