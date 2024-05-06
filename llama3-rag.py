from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, GPTQConfig
from time import time
from datetime import datetime
import transformers
import torch
from transformers import pipeline
from langchain.llms import HuggingFacePipeline
from huggingface_hub import login

hf_access_token_write_permission = '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'
hf_cache_dir = "D:/LLM/meta"
#model_id = 'meta-llama/Meta-Llama-3-8B-Instruct'
model_id = 'gaurav021201/Meta-Llama-3-8B-GPTQ'
login(token=hf_access_token_write_permission,
      add_to_git_credential=False,
      write_permission=True)
#loading a Quantized Model
quantization_config = GPTQConfig(bits=4,
                                 disable_exllama=True,
                                 use_cuda_fp16=False)
model = AutoModelForCausalLM.from_pretrained(model_id,
                                             quantization_config=quantization_config,
                                             device_map="auto")
print(model)
tokenizer = AutoTokenizer.from_pretrained(model_id)
text_generation_pipeline = pipeline("text-generation",
                                    model=model,
                                    tokenizer=tokenizer,
                                    torch_dtype=torch.float16,
                                    max_length=3000,
                                    device_map="auto")
llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
#loading llama3
"""model_config = AutoConfig.from_pretrained(model_id,
                                          trust_remote_code=True,
                                          max_new_tokens=1024,
                                          cache_dir=hf_cache_dir)
model = AutoModelForCausalLM.from_pretrained(model_id,
                                             trust_remote_code=True,
                                             config=model_config,
                                             device_map='auto',
                                             cache_dir=hf_cache_dir)
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=hf_cache_dir)
tokenizer.pad_token = tokenizer.eos_token
text_generation_pipeline = pipeline("text-generation",
                                    model=model,
                                    tokenizer=tokenizer,
                                    torch_dtype=torch.float16,
                                    max_length=3000,
                                    device_map="auto")
llm = HuggingFacePipeline(pipeline=text_generation_pipeline)"""
"""prompt = <|begin_of_text|>
            <|start_header_id|>
              user
            <|end_header_id|>
              Hello it is nice to meet you!
            <|eot_id|>
            <|start_header_id|>
              assistant
            <|end_header_id|>
         
out = llm.invoke(prompt)
print(out)"""
loader = PyPDFLoader("<<pdf file>>")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
embedding_function = SentenceTransformerEmbeddings(model_name="BAAI/bge-small-en-v1.5")
vectorstore = Chroma(collection_name="sample_collection", embedding_function=embedding_function)
vectorstore.add_documents(texts)
retriever = vectorstore.as_retriever(k=2)
print(texts)


class Pipeline:
    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever

    def retrieve(self, question):
        docs = self.retriever.invoke(question)
        return "\n\n".join([d.page_content for d in docs])

    def augment(self, question, context):
        return f"""
            <|begin_of_text|>
            <|start_header_id|>
              system
            <|end_header_id|>
               You are a helpful, respectful and honest assistant designated answer
               questions related to the user's document.If the user tries to ask out of 
               topic questions do not engage in the conversation.If the given context 
               is not sufficient to answer the question,Respond as "I am not aware".
            <|eot_id|>
            <|start_header_id|>
               user
            <|end_header_id|>
              Answer the user question based on the context provided below
              Context :{context}
              Question: {question}
            <|eot_id|>
            <|start_header_id|>
              assistant
            <|end_header_id|>"""

    def parse(self, string):
        return string.split("<|end_header_id|>")[-1]

    def generate(self, question):
        context = self.retrieve(question)
        prompt = self.augment(question, context)
        answer = self.llm.invoke(prompt)
        return self.parse(answer)


pipe = Pipeline(llm, retriever)
print(pipe)


def llama3_chat():
    print(
        "Hello!!!! I am llama3 and I can help with your document. "
        "\nIf you want to stop you can enter STOP at any point!")
    print()
    print("-------------------------------------------------------------------------------------")
    pipe_new = Pipeline(llm, retriever)
    question = input()
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    while question != "STOP":
        out = pipe_new.generate(question)
        print(out)
        print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        print("\nIs there anything else you would like my help with?")
        print("-------------------------------------------------------------------------------------")
        question = input()
        print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


llama3_chat()
