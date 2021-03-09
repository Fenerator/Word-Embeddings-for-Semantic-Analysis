#!/usr/bin/python

import sys
import string

def preprocessing(textfile):
    # with is like your try .. finally block in this case
    with open(textfile, 'r', encoding='utf8') as infile:
        lines = infile.readlines()

    for i in lines:
        i = i.split()
        i = [word.lower() for word in i] #set to lowercase
        # remove punctuation from each word
        table = str.maketrans('', '', string.punctuation)
        i = [word.translate(table) for word in i]
        print(i)




    with open('preprocessed_file', 'w', encoding='utf8') as outfile:
        outfile.writelines(lines)

    return True






def main(arguments):
    print('Number of arguments:', len(arguments), 'arguments.')
    print('Argument List:', arguments)

    textfile = arguments[0]
    B = arguments[1]
    T = arguments[2]

    preprocessing(textfile)




if __name__ == "__main__":
    if len(sys.argv) ==1:
       main(['text.txt', 'B.txt', 'T.txt'])
    else:
        main(sys.argv[1:])