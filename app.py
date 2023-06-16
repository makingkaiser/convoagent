import pinecone
import os
from langchain.agents import Tool
from langchain.agents import AgentType
from langchain.agents import initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA
#from langchain.chains import RetrievalQAWithSourcesChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone

from langchain.chat_models import ChatOpenAI

'Flask packages'
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin

os.environ['OPENAI_API_KEY'] = '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'

#Setting up ability to query Pinecone with tools

# Vectorstore Index
index_name = 'podcasts'

#Pincone API key
api_key = "721f43b4-3f81-4e8d-b42e-2a0eb86e5141"
env = "asia-southeast1-gcp-free"

pinecone.init(api_key=api_key, environment=env)


embeddings = OpenAIEmbeddings()
docsearch = Pinecone.from_existing_index(index_name,embeddings, namespace = 'NUSundergradfaq')
llm = ChatOpenAI(temperature=0)
embeddings_retriever = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever())
#podcast_retriever = RetrievalQAWithSourcesChain.from_chain_type(llm=retrieval_llm, chain_type="stuff", retriever=docsearch.as_retriever())


tools = [
    
    Tool(
        name = 'Knowledge Base',
        func=embeddings_retriever.run,
        description="Always consult this knowledge base for answers. Input should be a fully formed question.",
        return_direct=True
    )
]

memory = ConversationBufferMemory(memory_key="chat_history")

agent_chain = initialize_agent(tools, llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, memory=memory)


#agent_chain.run(input="Hello my name is Kaiser")
#agent_chain.run(input="What is my name?")

app = Flask(__name__)
CORS(app)
@app.route('/api/execute-gpt-query', methods=['POST'])
@cross_origin()

def execute_python_function():
    
    print("hello")
    print(request.json)
    input_data = request.json.get('input')
    print(input_data)

    output = agent_chain.run(input=input_data)
    return jsonify({'output': output})
if __name__ == '__main__':
    app.run()