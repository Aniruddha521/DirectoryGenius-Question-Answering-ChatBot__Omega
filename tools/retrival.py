from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

class retrival_question_answering():
    def __init__(self, 
                 name = "ReqriveQA",
                desc = "It is used to answer the questions based on the documents retrived from vector Database"):
        self.name = name
        self.desc = desc
    def run(
            self,
            llm,
            question: str,
            retriever,
            prompt: PromptTemplate = None,
            chain_type: str = "stuff",
    ):
        if chain_type == "stuff":
            kwargs = {"prompt": prompt}
        else:
            kwargs = {}

        qa_chain = RetrievalQA.from_chain_type(
            llm,
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs=kwargs,
            )

        return qa_chain({"query": question} )