text_list = ['war', 'es', 'god', 'something', 'war', 'something', 'states', 'united', 'something', 'health', 'something', 'fair', 'to', 'you', 'a', 'history', 'something', 'immigrant', 'something', 'everyone', 'as', 'me', 'god', 'something', 'war', 'something', 'states', 'war']

def count_occurrence(text_list, word): # P(list) for PPMI formula
    # delete all words not in word_list
    cleaned_text = [x for x in text_list if x == word]
    print('Cleaned: ', cleaned_text)
    result = len(cleaned_text)

    return result


print(count_occurrence(text_list, 'war'))
