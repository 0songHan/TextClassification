
id = 0
id2label = {}

with open ('./data/class.txt','r') as f :
    for line in f.readlines():
        line = line.strip('\n').strip()
        id2label[id] = line
        id+=1

print(id2label)