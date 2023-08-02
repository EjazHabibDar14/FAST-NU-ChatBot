from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import WebBaseLoader
from  langchain.schema import Document
from typing import Iterable


loader = WebBaseLoader( ['https://www.nu.edu.pk/',
    'https://www.nu.edu.pk/Degree-Programs',
    'https://www.nu.edu.pk/Admissions/Schedule',
    'https://www.nu.edu.pk/Admissions/HowToApply',
    'https://www.nu.edu.pk/Admissions/EligibilityCriteria',
    'https://www.nu.edu.pk/Admissions/Scholarship',
    'https://www.nu.edu.pk/Admissions/TestPattern',
    'https://www.nu.edu.pk/Admissions/FeeStructure',
    'http://isb.nu.edu.pk/',
    'http://isb.nu.edu.pk/Faculty/allfaculty#cs',
    'http://isb.nu.edu.pk/Faculty/allfaculty#ms',
    'http://isb.nu.edu.pk/Faculty/allfaculty#ee',
    'http://isb.nu.edu.pk/Faculty/allfaculty#sh',
    'http://isb.nu.edu.pk/Student/Grading',
    'https://nu.edu.pk/Student/Calender',
    'https://nu.edu.pk/Student/Conduct' ,
    'https://nu.edu.pk/Student/HECEquivalence',
    'https://nu.edu.pk/Student/FinancialRules',
    'https://nu.edu.pk/University/History',
    'https://nu.edu.pk/University/Foundation' ,
    'https://nu.edu.pk/University/Chancellor',
    'https://nu.edu.pk/vision-and-mission' ,
    'https://nu.edu.pk/University/Trustees' ,
    'https://nu.edu.pk/University/Governers' ,
    'https://nu.edu.pk/University/Officers' ,
    'https://nu.edu.pk/University/Headquarters',
    'https://nu.edu.pk/University/PhDFaculty' ,
    'https://nu.edu.pk/University/HECSupervisors',
    'https://nu.edu.pk/University/ExternalThesisReviewer' ,
    'https://nu.edu.pk/home/ContactUs'])

txt_file_as_loaded_docs = loader.load()
print(txt_file_as_loaded_docs)

splitter = CharacterTextSplitter(chunk_size=8000, chunk_overlap=500)
chunked_docs = splitter.split_documents(txt_file_as_loaded_docs)
print(chunked_docs)

def save_docs_to_jsonl(array:Iterable[Document], file_path:str)->None:
    with open(file_path, 'w') as jsonl_file:
        for doc in array:
            jsonl_file.write(doc.json() + '\n')

save_docs_to_jsonl(chunked_docs,"data.jsonl")