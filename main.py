from PyPDF2 import PdfReader
from factchecker import fact_check
from grammercheck import grammar, check_sentence
from coherence import keyword_cs, paragraph_cs, topicCheck
from logistic import regress
import csv
import os

filepath = "pdfs/"

def update_sample_data(pathway, filename, fact, key, para, gr, answer):
    # Step 1: Create a CSV file at the given path
    with open(pathway, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # Writing the header and some rows
        writer.writerow([filename, fact, key, para, gr, answer])

    print(f"sample_data.csv file updated")

def update_results(pathway, filename, answer, topic):
    # Step 1: Create a CSV file at the given path
    with open(pathway, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # Writing the header and some rows
        writer.writerow([filename, answer, topic])

    print(f"Result of {filename} added to results.csv")

for filename in os.listdir(filepath):
    name = "pdfs/"+filename
    reader = PdfReader(name)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    
    paragraph = text
    text=text.replace("\n"," ")
    
    fact = fact_check(text)
    key = keyword_cs(text)
    para = paragraph_cs(paragraph)
    gr = check_sentence(text)
    answer = regress(fact, key, para, gr)
    if answer==1:
        topic = topicCheck(text)
    else:
        topic = "NA"
    
    
    
    if __name__ == "__main__":
        # Set the pathway to the desired location for your CSV
        pathway = "data/sample_data.csv"
        path_result_csv = "data/results.csv"

        # Ensure the directory exists
        os.makedirs(os.path.dirname(pathway), exist_ok=True)

        # Create and save CSV file
        update_sample_data(pathway, filename, fact, key, para, gr, answer)
        update_results(path_result_csv, filename, answer, topic)
        
