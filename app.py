import streamlit as st
from llama_index import download_loader
import os
from pyvis.network import Network
import streamlit.components.v1 as components

# Set Google API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Download PDFReader
PDFReader = download_loader("PDFReader")

# Create the Gemini model
from llama_index.llms import Gemini
from llama_index.embeddings import GeminiEmbedding
llm = Gemini(model="models/gemini-pro", temperature=0, embedding=GeminiEmbedding,)

# Create the service context
from llama_index import ServiceContext
service_context = ServiceContext.from_defaults(llm=llm, chunk_size=512, embed_model=GeminiEmbedding)

# Create the graph store
from llama_index.graph_stores import SimpleGraphStore
graph_store = SimpleGraphStore()

# Create the storage context
from llama_index import StorageContext
storage_context = StorageContext.from_defaults(graph_store=graph_store)

# Create the Knowledge Graph Index
from llama_index import KnowledgeGraphIndex




# Streamlit app
def main():
    st.title("Knowledge Graph App")
    
    # File upload
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    
    if uploaded_file is not None:
        # Update the Knowledge Graph Index
        with st.spinner("Please wait while we generate your graph"):
            index = update_knowledge_graph_index(uploaded_file)
        st.snow()
        
        # Display the Knowledge Graph
        g = index.get_networkx_graph()
        net = Network(notebook=False, directed=True)
        net.from_nx(g)
        net.save_graph("knowledge_graph.html")
        HtmlFile = open("knowledge_graph.html", 'r', encoding='utf-8')
        source_code = HtmlFile.read() 
        st.markdown("## Knowledge Graph")
        components.html(source_code, width=800, height=600)
        #st.balloons()

        query = st.text_input("Enter your query")
        if st.button("Query"):
            with st.spinner("Please wait while we generate your response"):
                response = run_query(query=query, index=index)
        
        # Query the Knowledge Graph
        
            st.markdown(f"{response}", unsafe_allow_html=True)

# Function to update the Knowledge Graph Index
def update_knowledge_graph_index(uploaded_file):
    documents = PDFReader().load_data(file=uploaded_file)
    
    # Update the Knowledge Graph Index
    index = KnowledgeGraphIndex.from_documents(
        documents,
        max_triplets_per_chunk=1,
        storage_context=storage_context,
        service_context=service_context,
    )
    
    return index

def run_query(query, index):
    query_engine = index.as_query_engine(
    include_text=True,
    response_mode="tree_summarize",
                )
    resp = query_engine.query(query)
    response = llm.complete(f"You are a teching assistant, answer the question {query} with the keeping {resp} in mind, do not hallucinate. If the {resp} is not enough to answer the question, you can use your abilities along with the {resp} to answer the question. If you are not aware of the answer, you can answer it with your pretrained knowledge ")
    return response

    
if __name__ == "__main__":
    main()
