def count_occurrence(text_list, word_list): # P(list) for PPMI formula
    counts = [0] * len(word_list)  # stores values
    # delete all words not in word_list
    cleaned_text = [x for x in text_list if x in word_list]
    print('cleaned text ', cleaned_text)
    counts = Counter(cleaned_text)
    print('Counts: ', counts)

    # write counts into vector (at correct position)
    count_vector = []
    for el in word_list:
        count_vector.append(counts[el])
    print('Count Vector: ', count_vector)

    return count_vector