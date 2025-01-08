from transformers import T5TokenizerFast, T5Model
from torch.nn.functional import cosine_similarity
import torch
import spacy

spacy.prefer_gpu()

tokenizer = T5TokenizerFast.from_pretrained('t5-large')
model = T5Model.from_pretrained('t5-large')
nlp = spacy.load("en_core_sci_lg")
def keyword_cs(text):
    # Example text
    #text = """"""
    # Process the text
    print("Starting Coherence check...")
    doc = nlp(text)

    # Extract named entities
    #named_entities = [(ent.text, ent.label_) for ent in doc.ents]
    sentences = [ent.text for ent in doc.ents]
    #print("Named Entities:", named_entities)
    #print("Named Entities:", sentences)

    inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True, max_length=512)

    # Step 3: Get embeddings
    with torch.no_grad():
        outputs = model.encoder(**inputs)

    # Extract sentence embeddings by averaging token embeddings
    # outputs.last_hidden_state.shape: (batch_size, seq_length, hidden_size)
    sentence_embeddings = outputs.last_hidden_state.mean(dim=1)

    # Step 4: Calculate semantic coherence between consecutive sentences
    similarities = []
    for i in range(len(sentences)):  # Outer loop
        for j in range(i-1):  # Inner loop
            sim = cosine_similarity(sentence_embeddings[i], sentence_embeddings[j], dim=0).item()
            if(sim!=1):
                similarities.append((sim)*(sim)*(sim)*(sim)*(sim))
    # Step 5: Compute average coherence percentage
    average_similarity = sum(similarities) / len(similarities) if similarities else 1.0
    coherence_percentage = average_similarity*1000

    # Output the results
    print(f"Semantic Coherence Percentage: {coherence_percentage:.2f}%")
    return coherence_percentage
    #print(f"Individual Sentence Similarities: {similarities}")
    
def paragraph_cs(paragraph):

    # Paragraph to analyze
    #paragraph = """Sun is flat. Sun is round"""

    #paragraph = paragraph.replace('\n','  ')  # Clean up newlines

    # Step 1: Split paragraph into sentences
    print("Starting paragraph check...")
    sentences = paragraph.split(".\n")
    for i in range(len(sentences)):
        sentences[i] = sentences[i].replace('\n',' ')
    #print("Sentences:", sentences)

    # Step 2: Tokenize and encode sentences
    inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True, max_length=512)
    # Step 3: Get embeddings
    with torch.no_grad():
        outputs = model.encoder(**inputs)

    # Extract sentence embeddings by averaging token embeddings
    # outputs.last_hidden_state.shape: (batch_size, seq_length, hidden_size)
    sentence_embeddings = outputs.last_hidden_state.mean(dim=1)

    # Step 4: Calculate semantic coherence between consecutive sentences
    similarities = []
    for i in range(len(sentences)):
        for j in range(i - 1):
            sim = cosine_similarity(sentence_embeddings[i], sentence_embeddings[j], dim=0).item()
            similarities.append(sim)

    # Step 5: Compute average coherence percentage
    average_similarity = sum(similarities) / len(similarities) if similarities else 1.0
    coherence_percentage = average_similarity * 100

    # Output the results
    print(f"Semantic Paragraph Coherence Percentage: {coherence_percentage:.2f}%")
    return coherence_percentage
    #print(f"Individual Paragraph Similarities: {similarities}")
    
