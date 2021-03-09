def preprocessing(textfile):
    # with is like your try .. finally block in this case
    with open('preprocessed_file', 'w', encoding='utf8') as outfile:
        with open(textfile, 'r', encoding='utf8') as infile:
            lines = infile.readlines()

            for i in lines:
                i = i.split()
                i = [word.lower() for word in i]  # set to lowercase
                # remove punctuation from each word
                table = str.maketrans('', '', string.punctuation)
                i = [word.translate(table) for word in i]

                i = word +

                outfile.write(str(i))
                outfile.write('\n')
