import pandas as pd
text_list = ['wars', 'es', 'god', 'something', 'war', 'something', 'states', 'united', 'something', 'health', 'something', 'fair', 'to', 'you', 'a', 'history', 'something', 'immigrant', 'something', 'everyone', 'as', 'me', 'god', 'something', 'war', 'something', 'states', 'war']
'''
text_list[el-border:el]
text_list[el+1:el+border+1] '''


l = ['It', 'is', 'two', 'pm.']

data = {'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]}
data_df = pd.DataFrame(data, index=['a', 'b', 'c'])
print(data_df.transpose())

print(data_df)
