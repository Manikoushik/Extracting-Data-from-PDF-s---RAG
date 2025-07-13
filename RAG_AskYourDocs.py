from pydoc import doc
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage,AIMessage
import uuid

# Other modules and packages
import os
from dotenv import load_dotenv

load_dotenv()

GENAI_API_KEY = os.getenv("GENAI_API_KEY")

genai.configure(api_key=GENAI_API_KEY)
llm = genai.GenerativeModel("gemini-2.5-pro")

#response = llm.generate_content("Tell me a joke about cats")
#print(response.text)

#Loading PDF document

loader = PyPDFLoader('data\Oppenheimer-2006-Applied_Cognitive_Psychology.pdf')
pages = loader.load()

#Split document
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500,
                                              chunk_overlap=200,
                                              length_function=len,
                                              separators=["\n\n","\n"," "])
chunks = text_splitter.split_documents(pages)

#Create Embeddings

def get_embedding_function():
  embedding_model = 'models/embedding-001'
  embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model, google_api_key=GENAI_API_KEY,task_type="SEMANTIC_SIMILARITY")
  return embeddings

embedding_function = get_embedding_function()

#Chroma Database from the documents to store the chunks
def create_vectorstore(chunks, embedding_function, vectorstore_path):
    #Create a list of unique ids for each document based on the content
    ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, doc.page_content)) for doc in chunks]
    
    # Ensure that only unique docs with unique ids are kept
    unique_ids = set()
    unique_chunks = []
    
    unique_chunks = [] 
    for chunk, id in zip(chunks, ids):     
        if id not in unique_ids:       
            unique_ids.add(id)
            unique_chunks.append(chunk) 

    # Create a new Chroma database from the documents
    vectorstore = Chroma.from_documents(documents=unique_chunks, 
                                        ids=list(unique_ids),
                                        embedding=embedding_function, 
                                        persist_directory = vectorstore_path)

    return vectorstore

#Create Vectorstore
vectorstore = create_vectorstore(chunks=chunks, embedding_function=embedding_function,vectorstore_path="vectorstore")

#load Vectorstore
load_vectorstore = Chroma(persist_directory="vectorstore", embedding_function=embedding_function)

#Create Retriever and get relevent chunks
retriever = load_vectorstore.as_retriever(search_type="similarity")
relevant_chunks = retriever.invoke("What is the title of the article?")
# print(relevant_chunks)


# Prompt template
PROMPT_TEMPLATE = """
  You are an assistant for question-answering tasks.
  Use the following pieces of retrieved context to answer
  the question. If you don't know the answer, say that you
  don't know. DON'T MAKE UP ANYTHING.

  {context}

  ---

  Answer the question based on the above context: {question}
  """

#concatenate context text
context_text = "\n\n--\n\n".join([doc.page_content for doc in relevant_chunks])

#create prompt
prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
prompt = prompt_template.format(context=context_text,question="What is the title of the article?")

llm.generate_content(prompt)
#print(llm.generate_content(prompt).text)

def format_docs(docs):
  return "\n\n".join(doc.page_content for doc in docs)

def invoke_gemini_native(input_dict):
  messages = input_dict.to_messages()
  
  gemini_messages_format = []
  for msg in messages:
    if isinstance(msg, HumanMessage):
      gemini_messages_format.append({'role': 'user', 'parts': [msg.content]})
    elif isinstance(msg, AIMessage):
      gemini_messages_format.append({'role': 'model', 'parts': [msg.content]})
  
  response = llm.generate_content(gemini_messages_format)

  return response.text

llm_runnable = RunnableLambda(invoke_gemini_native)


#Generate Structured Response
rag_chain = (
            {"context":retriever | format_docs, "question": RunnablePassthrough()}
            | prompt_template | llm_runnable)


# --- Interactive Query Loop ---
print("Welcome to the Document Q&A system!")
print("Enter your query to get information from the document.")
print("Type 'E' to exit or 'C' to continue with another query.")

while True:
    user_input = input("\nEnter your query: ").strip()

    if user_input.upper() == 'E':
        print("Exiting the Q&A system. Goodbye!")
        break
    elif user_input.upper() == 'C':
        # If 'C' is entered initially, just prompt for the next query
        continue 
    else:
        try:
            answer = rag_chain.invoke(user_input)
            print("\n--- Answer ---")
            print(answer)
            print("--------------")
        except Exception as e:
            print(f"An error occurred while processing your query: {e}")
    
    # After providing an answer (or handling an error), ask for next action
    action = input("Enter 'E' to exit or 'C' to continue with another query: ").strip()
    if action.upper() == 'E':
        print("Exiting the Q&A system. Goodbye!")
        break
    elif action.upper() == 'C':
        continue
    else:
        print("Invalid input. Continuing with the next query by default.")