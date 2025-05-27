import os
import chainlit as cl
from dotenv import load_dotenv
from operator import itemgetter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.runnable.config import RunnableConfig
from tqdm.asyncio import tqdm_asyncio
import asyncio
from tqdm.asyncio import tqdm

# GLOBAL SCOPE - ENTIRE APPLICATION HAS ACCESS TO VALUES SET IN THIS SCOPE #
# ---- ENV VARIABLES ---- #
"""
This function will load our environment file (.env) if it is present.

NOTE: Make sure that .env is in your .gitignore file - it is by default, but please ensure it remains there.
"""
load_dotenv()

"""
We will load our environment variables here.
"""
HF_LLM_ENDPOINT = os.environ["HF_LLM_ENDPOINT"]
HF_EMBED_ENDPOINT = os.environ["HF_EMBED_ENDPOINT"]
HF_TOKEN = os.environ["HF_TOKEN"]

# ---- GLOBAL DECLARATIONS ---- #

# -- RETRIEVAL -- #
"""
1. Load Documents from Text File
2. Split Documents into Chunks
3. Load HuggingFace Embeddings (remember to use the URL we set above)
4. Index Files if they do not exist, otherwise load the vectorstore
"""
### 1. CREATE TEXT LOADER AND LOAD DOCUMENTS
### NOTE: PAY ATTENTION TO THE PATH THEY ARE IN.
text_loader = TextLoader("./data/paul_graham_essays.txt")
documents = text_loader.load()

### 2. CREATE TEXT SPLITTER AND SPLIT DOCUMENTS
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=30)
split_documents = text_splitter.split_documents(documents)

### 3. LOAD HUGGINGFACE EMBEDDINGS
hf_embeddings = HuggingFaceEndpointEmbeddings(
    model=HF_EMBED_ENDPOINT,
    task="feature-extraction",
    huggingfacehub_api_token=HF_TOKEN,
)


async def add_documents_async(vectorstore, documents):
    """Add documents asynchronously to an existing vectorstore.

    Args:
        vectorstore: The FAISS vectorstore to add documents to.
        documents: List of documents to be added to the vectorstore.

    Returns:
        None
    """
    await vectorstore.aadd_documents(documents)


async def process_batch(vectorstore, batch, is_first_batch, pbar):
    """Process a batch of documents for vectorstore creation or addition.

    This function either creates a new vectorstore from the first batch of documents
    or adds documents to an existing vectorstore for subsequent batches.

    Args:
        vectorstore: Existing FAISS vectorstore (None for first batch).
        batch: List of documents to process in this batch.
        is_first_batch (bool): True if this is the first batch (creates new vectorstore),
                              False if adding to existing vectorstore.
        pbar: Progress bar object to update with batch processing progress.

    Returns:
        FAISS vectorstore: Either newly created or existing vectorstore with added documents.
    """
    if is_first_batch:
        result = await FAISS.afrom_documents(batch, hf_embeddings)
    else:
        await add_documents_async(vectorstore, batch)
        result = vectorstore
    pbar.update(len(batch))
    return result


async def main():
    """Create and populate a FAISS vectorstore from document batches.

    This function processes documents in batches to create a vectorstore for retrieval.
    The first batch initializes the vectorstore, while subsequent batches are processed
    in parallel to add documents efficiently.

    Returns:
        Retriever: A FAISS retriever object for document search and retrieval.

    Raises:
        Various exceptions from FAISS operations or document processing.
    """
    print("Indexing Files")

    vectorstore = None
    batch_size = 32

    batches = [
        split_documents[i : i + batch_size]
        for i in range(0, len(split_documents), batch_size)
    ]

    async def process_all_batches():
        nonlocal vectorstore
        tasks = []
        pbars = []

        for i, batch in enumerate(batches):
            pbar = tqdm(
                total=len(batch), desc=f"Batch {i+1}/{len(batches)}", position=i
            )
            pbars.append(pbar)

            if i == 0:  # first batch is processed directly to initialize vectorstore
                vectorstore = await process_batch(None, batch, True, pbar)
            else:  # the remaining batches are processed in parallel
                tasks.append(process_batch(vectorstore, batch, False, pbar))

        if tasks:
            await asyncio.gather(*tasks)

        for pbar in pbars:
            pbar.close()

    await process_all_batches()

    hf_retriever = vectorstore.as_retriever()
    print("\nIndexing complete. Vectorstore is ready for use.")
    return hf_retriever


async def run():
    """Execute the main indexing process and return the retriever.

    This is a wrapper function that calls the main() function to create
    the vectorstore and returns the resulting retriever.

    Returns:
        Retriever: A FAISS retriever object for document search and retrieval.
    """
    retriever = await main()
    return retriever


hf_retriever = asyncio.run(run())

# -- AUGMENTED -- #
"""
1. Define a String Template
2. Create a Prompt Template from the String Template
"""
### 1. DEFINE STRING TEMPLATE
RAG_PROMPT_TEMPLATE = """\
<|start_header_id|>system<|end_header_id|>
You are a helpful assistant. You answer user questions based on provided context. If you can't answer the question with the provided context, say you don't know.<|eot_id|>

<|start_header_id|>user<|end_header_id|>
User Query:
{query}

Context:
{context}<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>
"""

### 2. CREATE PROMPT TEMPLATE
rag_prompt = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

# -- GENERATION -- #
"""
1. Create a HuggingFaceEndpoint for the LLM
"""
### 1. CREATE HUGGINGFACE ENDPOINT FOR LLM
hf_llm = HuggingFaceEndpoint(
    endpoint_url=f"{HF_LLM_ENDPOINT}",
    task="text-generation",
    max_new_tokens=512,
    top_k=10,
    top_p=0.95,
    typical_p=0.95,
    temperature=0.01,
    repetition_penalty=1.03,
)


@cl.author_rename
def rename(original_author: str):
    """Rename the author of messages in the chat interface.

    This function customizes the display name of message authors in the Chainlit
    chat interface. It maps the default 'Assistant' author to a more specific
    'Paul Graham Essay Bot' name.

    Args:
        original_author (str): The original author name to be renamed.

    Returns:
        str: The new author name, or the original name if no mapping exists.
    """
    rename_dict = {"Assistant": "Paul Graham Essay Bot"}
    return rename_dict.get(original_author, original_author)


@cl.on_chat_start
async def start_chat():
    """Initialize a new chat session with the RAG chain.

    This function is called at the start of every user session. It builds
    an LCEL (LangChain Expression Language) RAG chain that combines document
    retrieval with language model generation, and stores it in the user session
    for subsequent message handling.

    The RAG chain includes:
    - Document retrieval using the FAISS vectorstore
    - Prompt formatting with retrieved context
    - Text generation using the HuggingFace LLM

    Returns:
        None: The chain is stored in the user session rather than returned.
    """

    ### BUILD LCEL RAG CHAIN THAT ONLY RETURNS TEXT
    lcel_rag_chain = (
        {"context": itemgetter("query") | hf_retriever, "query": itemgetter("query")}
        | rag_prompt
        | hf_llm
    )

    cl.user_session.set("lcel_rag_chain", lcel_rag_chain)


@cl.on_message
async def main(message: cl.Message):
    """Handle incoming chat messages and generate RAG-based responses.

    This function is called every time a message is received from a user session.
    It uses the LCEL RAG chain stored in the user session to generate a response
    by retrieving relevant documents and generating contextual answers.

    The response is streamed back to the user in real-time as tokens are generated
    by the language model.

    Args:
        message (cl.Message): The incoming message object containing user input.

    Returns:
        None: The response is sent directly to the chat interface.

    Raises:
        Various exceptions from the RAG chain execution or streaming process.
    """
    lcel_rag_chain = cl.user_session.get("lcel_rag_chain")

    msg = cl.Message(content="")

    async for chunk in lcel_rag_chain.astream(
        {"query": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    await msg.send()
