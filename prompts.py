from langchain.prompts import ChatPromptTemplate, PromptTemplate

template_1 = (
                "Combine the chat history and follow up question into "
                "a standalone question. Chat History: {chat_history}"
                "Follow up question: {question}"
            )
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(template_1)

template_2 = (
                "Combine the human asked question and AI generated answer "
                "and create title for the chat topic."
                "Remember it should be of maximium five tokens"
                "Human asked question: {question}"
                "AI generated reply: {reply}"
            )
CONDENSE_title_PROMPT = PromptTemplate.from_template(template_2)


template_3 = """Use the following pieces of context to answer the question at the end. \
    If you don't know the answer, just say that you don't know, don't try to make up an answer.\
    Also provide any example or code along with answer for better understanding. \
    Also remember to write the code part in given format:
    ```bash
        write code here
    ```
    Always encourage the user to ask if they still have any query at the end of the answer.
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = ChatPromptTemplate.from_template(template_3)