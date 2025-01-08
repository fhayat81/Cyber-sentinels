import language_tool_python
tool = language_tool_python.LanguageTool('en-US')

def grammar(text):
    print("Starting grammer check...")
    

    matches = tool.check(text)

    # Calculate correctness
    total_words = len(text.split())
    total_errors = len(matches)
    correct_percentage = ((total_words - total_errors) / total_words) * 100

    print(f"Total Words: {total_words}")
    print(f"Total Errors: {total_errors}")
    print(f"Grammatical Correctness: {correct_percentage:.2f}%")
    
def check_sentence(sentence):
    # Check for errors using LanguageTool
    matches = tool.check(sentence)

    # Categorize errors
    grammar_issues = []
    punctuation_issues = []
    completeness_issues = []

    for match in matches:
        # Identify types of issues based on rule category
        if "Grammar" in match.category:
            grammar_issues.append(match)
        elif "Punctuation" in match.category:
            punctuation_issues.append(match)
        else:
            completeness_issues.append(match)

    # Custom completeness check
    ends_correctly = sentence.strip().endswith(('.', '!', '?'))
    if not ends_correctly:
        completeness_issues.append("Sentence does not end with proper punctuation.")

    # Normalize error counts
    total_issues = len(grammar_issues) + len(punctuation_issues) + len(completeness_issues)
    max_issues = max(len(sentence.split()), 10)  # Normalize against word count or max threshold

    # Calculate cumulative error percentage
    error_percentage = (total_issues / max_issues) * 100

    # Results
    print(f"Percentage error in grammer: {error_percentage}")
    return error_percentage
    # print( {
    #     "Grammar Issues": len(grammar_issues),
    #     "Punctuation Issues": len(punctuation_issues),
    #     "Completeness Issues": len(completeness_issues),
    #     "Cumulative Error Percentage": min(error_percentage, 100),  # Cap at 100%
    #     "Details": {
    #         "Grammar": grammar_issues,
    #         "Punctuation": punctuation_issues,
    #         "Completeness": completeness_issues
    #     }
    # })
# Check each sentence
