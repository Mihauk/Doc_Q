import os 
import torch
import textwrap

#importing llm(will require API key. If using local llm you can comment these lines)
from langchain.llms import GooglePalm
from langchain.chat_models import ChatOpenAI


# Other langchain imports for loading documents, spliting them into chunks and using for retrival
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader


from langchain.indexes import VectorstoreIndexCreator

'''
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
'''

#importing vector embeddings tthat you want to use
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings import GooglePalmEmbeddings
from langchain.embeddings import HuggingFaceInstructEmbeddings

#importing vectorstores to use as index
from langchain.vectorstores import Chroma
from langchain.vectorstores import FAISS

#importing the GUI Library
import panel as pn

device = "cpu"
if torch.cuda.is_available():
  device = "cuda"


pn.extension('texteditor', template="bootstrap", sizing_mode='stretch_width')
pn.state.template.param.update(
    main_max_width="690px",
    header_background="#F08080",
)

select_llm = pn.widgets.RadioButtonGroup(
    name='llm', 
    options=['ChatOpenAI', 'GooglePalm'],
    value='GooglePalm'
)

apikey = pn.widgets.PasswordInput(
    value="", placeholder="Enter your API Key here...", width=300
)


prompt = pn.widgets.TextEditor(
    value="", placeholder="Enter your questions here...", height=160, toolbar=False
)
run_button = pn.widgets.Button(name="Run!")


select_k = pn.widgets.IntSlider(
    name="Number of relevant chunks", start=1, end=5, step=1, value=2
)

select_chain_type = pn.widgets.RadioButtonGroup(
    name='Chain type', 
    options=['stuff', 'map_reduce', "refine", "map_rerank"],
    value='map_reduce'
)

widgets = pn.Row(
    pn.Column(prompt, run_button, margin=5),
    pn.Card(
        "Chain type:",
        pn.Column(select_chain_type, select_k),
        title="Advanced settings"
    ), width=670
)

#methods for nicely printing out llm's response
def wrap_text_preserve_newlines(text, width=110):
    # Split the input text into lines based on newline characters
    lines = text.split('\n')

    # Wrap each line individually
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

    # Join the wrapped lines back together using newline characters
    wrapped_text = '\n'.join(wrapped_lines)

    return wrapped_text

'''
def process_llm_response(llm_response):
    print(wrap_text_preserve_newlines(llm_response['result']))
    print('\nSources:')
    for source in llm_response["source_documents"]:
        print(source.metadata['source'])
'''


def qa(query, llm, chain_type, k):
    # load document
    loader = DirectoryLoader(f'./Docs', glob="./*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    
    # split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
                                           chunk_size=1000,
                                           chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    
    # select which embeddings we want to use
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl",
                                                      model_kwargs={"device": device})
    #embeddings = OpenAIEmbeddings(model_kwargs={"device": device})
    #embeddings = GooglePalmEmbeddings(model_kwargs={"device": device})
    
    # create the vectorestore to use as the index
    #db = Chroma.from_documents(texts, embeddings)
    db = FAISS.from_documents(texts, embeddings)
    
    # expose this index in a retriever interface
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})

    # create a chain to answer questions 
    qa_chain = RetrievalQA.from_chain_type(
    llm=llm, chain_type=chain_type, retriever=retriever, return_source_documents=True)
    llm_response = qa_chain({"query": query})
    #print(result['result'])
    return llm_response


convos = []  # store all panel objects in a list

def qa_result(_):
    os.environ['GOOGLE_API_KEY' if select_llm.value=="GooglePalm" else 'OPENAI_API_KEY'] = apikey.value
    prompt_text = prompt.value
    if prompt_text:
        response = qa(query=prompt_text, llm =GooglePalm(temperature = 0.1) if select_llm.value=="GooglePalm" else ChatOpenAI(temperature = 0.1), chain_type=select_chain_type.value, k=select_k.value)
        convos.extend([
            pn.Row(
                pn.panel("\U0001F60A", width=10),
                prompt_text,
                width=600
            ),
            pn.Row(
                pn.panel("\U0001F916", width=10),
                pn.Column(
                    wrap_text_preserve_newlines(response['result']),
                    "Relevant source:",
                    pn.pane.Markdown("\n--------------\n".join(doc.metadata['source'] for doc in response["source_documents"]))
                )
            )
        ])
    return pn.Column(*convos, margin=15, width=575, min_height=400)


qa_interactive = pn.panel(
    pn.bind(qa_result, run_button),
    loading_indicator=True,
)

output = pn.WidgetBox('*Output will show up here:*', qa_interactive, width=670, scroll=True)

# layout
pn.Column(
    pn.pane.Markdown("""
    ## Question Answering with your PDF file
    
    1) Put all the PDF files in Docs folder. 2) Enter the API key.
    """),
    pn.Row(select_llm, apikey),
    widgets,
    output

).servable()
