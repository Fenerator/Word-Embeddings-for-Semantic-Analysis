l1 = ['a', 'b', 'c', 'd', 'c']
l2 = ['c', 'd']

l3 = []

l3.extend((l2))
l3.extend(l1)
print(l3)