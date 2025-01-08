from PyPDF2 import PdfReader
from factchecker import fact_check
from grammercheck import grammar, check_sentence
from coherence import keyword_cs, paragraph_cs
import csv
import os

filepath = "pdfs/"

def create_csv(pathway, filename, fact, key, para, gr):
    # Step 1: Create a CSV file at the given path
    with open(pathway, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # Writing the header and some rows
        writer.writerow([filename, fact, key, para, gr])

    print(f"CSV file created at: {pathway}")


def parse_csv(pathway):
    # Step 2: Parse the CSV file at the given path
    if not os.path.exists(pathway):
        print(f"Error: {pathway} does not exist.")
        return

    with open(pathway, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)

        # Reading and printing data row by row
        parsed_data = [row for row in reader]
        print("\nParsed CSV Data:")
        for row in parsed_data:
            print(row)

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
    
    if __name__ == "__main__":
        # Set the pathway to the desired location for your CSV
        pathway = "data/sample_data.csv"

        # Ensure the directory exists
        os.makedirs(os.path.dirname(pathway), exist_ok=True)

        # Create and save CSV file
        create_csv(pathway, filename, fact, key, para, gr)

        # Parse and print the CSV data
        parse_csv(pathway)