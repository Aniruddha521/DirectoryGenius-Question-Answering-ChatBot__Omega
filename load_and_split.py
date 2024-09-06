from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from langchain.document_loaders import TextLoader, BSHTMLLoader, UnstructuredMarkdownLoader, CSVLoader, PyPDFLoader
# from PyPDF2 import PdfReader, PdfFileWriter, PageObject

from custom_loader import ipynb_to_mardown
from collections import Counter


CHUNCK_SIZE = 1024
OVERLAP_SIZE = 200

extensions = lambda x: dict(Counter((lambda y: list(map(lambda z: z.split("/")[-1].split(".")[-1], y)))(x)))


python_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.PYTHON, chunk_size=CHUNCK_SIZE, chunk_overlap=OVERLAP_SIZE)
go_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.GO, chunk_size=CHUNCK_SIZE, chunk_overlap=OVERLAP_SIZE)
cpp_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.CPP, chunk_size=CHUNCK_SIZE, chunk_overlap=OVERLAP_SIZE)
java_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.JAVA, chunk_size=CHUNCK_SIZE, chunk_overlap=OVERLAP_SIZE)
js_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.JS, chunk_size=CHUNCK_SIZE, chunk_overlap=OVERLAP_SIZE)
php_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.PHP, chunk_size=CHUNCK_SIZE, chunk_overlap=OVERLAP_SIZE)
rupy_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.RUBY, chunk_size=CHUNCK_SIZE, chunk_overlap=OVERLAP_SIZE)
rust_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.RUST, chunk_size=CHUNCK_SIZE, chunk_overlap=OVERLAP_SIZE)
scala_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.SCALA, chunk_size=CHUNCK_SIZE, chunk_overlap=OVERLAP_SIZE)
swift_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.SWIFT, chunk_size=CHUNCK_SIZE, chunk_overlap=OVERLAP_SIZE)
md_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.MARKDOWN, chunk_size=CHUNCK_SIZE, chunk_overlap=OVERLAP_SIZE)
html_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.HTML, chunk_size=CHUNCK_SIZE, chunk_overlap=OVERLAP_SIZE)
latex_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.LATEX, chunk_size=CHUNCK_SIZE, chunk_overlap=OVERLAP_SIZE)
rst_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.RST, chunk_size=CHUNCK_SIZE, chunk_overlap=OVERLAP_SIZE)
proto_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.PROTO, chunk_size=CHUNCK_SIZE, chunk_overlap=OVERLAP_SIZE)
_default_text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNCK_SIZE, chunk_overlap=OVERLAP_SIZE)

text_loader = lambda x: TextLoader(x, autodetect_encoding=True).load()
ipynb_loader = lambda x: ipynb_to_mardown(TextLoader(x, autodetect_encoding=True).load())
pdf_loader = lambda x: PyPDFLoader(x).load()
Unstructured_Markdown_Loader = lambda x: UnstructuredMarkdownLoader(x).load()
HTML_Loader = lambda x: BSHTMLLoader(x).load()
_default_text_loader = lambda x: TextLoader(x, autodetect_encoding=True).load()

Default_File_Splitters = {
    "py": python_splitter,
    "ipynb": md_splitter,
    "go": go_splitter,
    "cpp": cpp_splitter,
    "c": cpp_splitter,
    "h": cpp_splitter,
    "java": java_splitter,
    "jar": java_splitter,
    "js": js_splitter,
    "ts": js_splitter,
    "php": php_splitter,
    "rjs": rupy_splitter,
    "rs": rust_splitter,
    "sc": scala_splitter,
    "swift": swift_splitter,
    "md": md_splitter,
    "html": html_splitter,
    "tex": latex_splitter,
    "rst": rst_splitter,
    "proto": proto_splitter,
}

Default_File_Loaders = {
    "py": text_loader,
    "ipynb": ipynb_loader,
    "go": text_loader,
    "cpp": text_loader,
    "c": text_loader,
    "h": text_loader,
    "java": text_loader,
    "jar": text_loader,
    "js": text_loader,
    "ts": text_loader,
    "php": text_loader,
    "rjs": text_loader,
    "rs": text_loader,
    "sc": text_loader,
    "swift": text_loader,
    "md": Unstructured_Markdown_Loader,
    "html": HTML_Loader,
    "tex": text_loader,
    "rst": text_loader,
    "proto": text_loader,
    "pdf" : pdf_loader
}
