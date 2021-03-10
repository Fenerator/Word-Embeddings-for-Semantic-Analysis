import numpy as np
def get_cosine_similarity(list1, list2):
    #convert to np array
    v1 = np.array(list1)
    v2 = np.array(list2)

    #normalize vectors
    v1 =  v1 / np.sqrt(np.sum(v1**2))
    v2 = v2 / np.sqrt(np.sum(v2**2))

    #calculate scalar product
    cosine_sim = np.dot(v1, v2)

    return cosine_sim

print(get_cosine_similarity([1.5640641494898921, 1.594321970582573, 1.6481284142783668, 0.0, 0.0, 0.0, 1.8194968325903478, 0.0, 0.0, 0.0, 0.0], [0.0, 2.464065584631231, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
#fut, hist
#print(get_cosine_similarity([3.3923174227787602, 0.0, 0.0], [0.0, 0.0, 4.807354922057604]))

