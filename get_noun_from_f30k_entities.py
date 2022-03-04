from flickr30k_entities.flickr30k_entities_utils import *
from collections import defaultdict
from stanfordcorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP(r'../stanford-corenlp-4.3.2')#, lang='de')

with open('train_val_test2016.en','r') as f:
    sentences = f.readlines()
sentence_list = []
for i in sentences:
    sentence_list.append(i.strip())

with open('train_val_test2016.txt','r') as f:
    image_names = f.readlines()
name_list = []
for i in image_names:
    name_list.append(i.strip())

noun = defaultdict(int)
nouns = defaultdict(int)

if __name__ == "__main__":
    for index in range(len(name_list)):
        image = name_list[index]
        sentence = sentence_list[index]
        x = get_sentence_data('../flickr30k_entities/Sentences/'+image.split('.')[0]+'.txt')
        flag = True
        for j in x:	# all matched
            if j['sentence'].replace(' ','').replace('”','"').replace('`', '\'').replace('"', '') == sentence.replace('"', '').replace(' ', ''):
                for t in j['phrases']:
                    phrase = t['phrase']#.lower()
                    try:
                        phrase_pos = nlp.pos_tag(phrase)
                    except:
                        print(phrase)
                    for pos in phrase_pos:
                        if pos[1] == 'NN':
                            noun[pos[0]] += 1
                        elif pos[1] == 'NNS':
                            nouns[pos[0]] += 1
                flag = False
                break
        if flag:
            print(sentence)
            for j in x:
                print(j['sentence'].lower())
            print()

    print(len(noun))
    print(len(nouns))
