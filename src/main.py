from langchain_community.llms import Ollama
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import warnings
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

warnings.filterwarnings("ignore")

# 1. 文档处理流水线
def process_documents():
    loader = WebBaseLoader("https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/")
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=14,
        separators=["\n\n", "\n", " ", ".", ",", "", ";", "!", "?"],
        length_function=len,
    )
    return text_splitter.split_documents(documents)

# 2. 本地模型初始化
def init_models():
    # 初始化 Embedding 模型
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # 初始化本地大模型
    llm = Ollama(
        model="qwen2.5-coder:14b",
        temperature=0.5,
    )
    
    return embeddings, llm

# 3. RAG pipeline 构建
def build_rag_pipeline(docs, embeddings, llm):
    # 创建向量存储
    vector_store = FAISS.from_documents(docs, embeddings)
    
    # 创建检索器
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    
    # 定义提示模板
    prompt_template = ChatPromptTemplate.from_template("""使用以下上下文片段回答问题：
    {context}
    
    问题：{question}
    答案：""")
    
    # 定义格式化上下文的函数
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    # 构建LCEL RAG链
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )
    
    # 包装链以返回源文档
    def invoke_with_sources(query):
        docs = retriever.invoke(query)
        result = rag_chain.invoke(query)
        return {"result": result, "source_documents": docs}
    
    return invoke_with_sources

if __name__ == "__main__":
    
    # 初始化组件
    split_docs = process_documents()
    embedding_model, llm_model = init_models()
    
    # 构建RAG系统
    qa_chain = build_rag_pipeline(split_docs, embedding_model, llm_model)
    
    # 创建Rich控制台
    console = Console()
    
    # 交互问答
    while True:
        query = input("\n请输入问题（输入q退出）: ")
        if query.lower() == 'q':
            break
            
        result = qa_chain(query)
        
        # 使用Rich美化输出
        answer_text = Text(result['result'])
        console.print(Panel(answer_text, title="📝 回答", border_style="green"))
        
        # 处理来源文档
        sources_content = ""
        for i, doc in enumerate(result['source_documents']):
            doc_content = doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
            sources_content += f"\n[bold]来源 {i+1}:[/bold]\n"
            sources_content += f"  - 内容片段: {doc_content}\n"
            if hasattr(doc, 'metadata') and doc.metadata:
                sources_content += f"  - 元数据: {doc.metadata}\n"
        
        console.print(Panel(sources_content, title="📚 参考来源", border_style="blue"))
