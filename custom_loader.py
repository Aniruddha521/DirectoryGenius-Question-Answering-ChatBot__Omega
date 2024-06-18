from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.documents.base import Document

def add_str(l):
    result = ""
    for i in l:
        result = result + i
    return result

def ipynb_to_mardown(ipynb_content: list):

    file_template_part1 = """The below given text is extracted from a\
                            .ipynb file in which the code written\
                            in the code cells is written as follows:\n
                        """

    file_template_part2 =  """```python\n# code in python\n```\n"""

    cells_schema = ResponseSchema(name="cells",
                             description="It contains a list of \
                                dictionary and each dictionary \
                                    contains  information of different cells")
    response_schemas = [cells_schema]
    parser = StructuredOutputParser.from_response_schemas(response_schemas)

    for content in ipynb_content:
        parse_content = parser.parse(content.page_content)
        file_content = ""
        for cell in parse_content["cells"]:
            if cell["cell_type"] == "markdown":
                file_content = file_content + "\n" + add_str(cell["source"])
            else:
                python_template = "```python" + "\n" + add_str(cell['source']) + "\n" + "```"
                file_content = file_content + "\n\n\n" + python_template + "\n"
        

        file_template_part3 =  f"""Analyse the text carefully and answer\
                                any question asked by the user correctly\
                                with full explanation of each step with\
                                revelant examples:\n------------------------------------------\n
                                    {file_content}"""
        
        file_template = file_template_part1 + "\n" + file_template_part2 + "\n" + file_template_part3

    return [Document(file_template)]