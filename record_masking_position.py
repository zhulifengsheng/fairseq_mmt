from stanfordcorenlp import StanfordCoreNLP
from tqdm import tqdm
# /path/stanford-corenlp-4.3.2
nlp = StanfordCoreNLP('../stanford-corenlp-4.3.2')#, lang='de')
# init
count = 0

out = open('data/masking/multi30k.masking.en', 'w', encoding='utf-8')
out_all_noun = open('data/masking/noun.en', 'w', encoding='utf-8')
out_all_nouns = open('data/masking/nouns.en', 'w', encoding='utf-8')

out_people = open('data/masking/multi30k.people.position', 'w', encoding='utf-8')
out_color = open('data/masking/multi30k.color.position', 'w', encoding='utf-8')
out_noun = open('data/masking/multi30k.noun.position', 'w', encoding='utf-8')
out_nouns = open('data/masking/multi30k.nouns.position', 'w', encoding='utf-8')

people = ['man', 'woman', 'boy', 'girl', 'people', 'men']
colors = ['orange', 'green', 'red', 'white', 'black', 'pink', 'blue', 'purple', 'tan', 'grey', 'gray', 'yellow', 'gold', 'golden', 'dark', 'brown', 'silver']
nouns = set()
noun = set()

num_p = 0
num_c = 0
num_n = 0
num_ns = 0

# multi30k.en consists of train+valid+test2016+test2017+testcoco
path_multi30k='data/multi30k/multi30k.en'

def check(_str):
    if '#' not in _str and '-' not in _str and '&' not in _str and len(_str) > 1 and _str not in people and _str not in colors:
        return True
    return False

# noun & nouns
with open(path_multi30k, 'r', encoding='utf-8') as f:
    for sentence in tqdm(f):
        sentence = sentence.strip()
        count += 1
        if '%' in sentence or '.a' in sentence: # fliter several sentences
            out_noun.write('-1\n')
            out_nouns.write('-1\n')
            continue
        
        _l = nlp.pos_tag(sentence)
        
        # fix 
        if len(_l) > len(sentence.split()):
            tmp = []
            tmp_idx = 0
            x = sentence.split()
            for idx, i in enumerate(x):
                if i == _l[tmp_idx][0]:
                    tmp.append(_l[tmp_idx])
                    tmp_idx += 1
                else:
                    str1 = _l[tmp_idx][0]
                    tmp_idx += 1
                    while str1 != i:
                        str1 += _l[tmp_idx][0]
                        tmp_idx += 1
                    tmp.append((i, 'UNK'))
            _l = tmp
        
        assert len(_l) == len(sentence.split()), 'error'
        
        flag_noun = 0
        flag_nouns = 0
        for idx, i in enumerate(_l):
            if i[-1] == 'NN' and i[0][-3:] != 'ing' and check(i[0]):
                out_noun.write(str(idx)+' ')
                flag_noun = 1
                noun.add(i[0])
                num_n += 1
            elif i[-1] == 'NNS' and check(i[0]):
                out_nouns.write(str(idx)+' ')
                flag_nouns = 1
                nouns.add(i[0])
                num_ns += 1
                
        if flag_noun == 0:
            out_noun.write(str(-1))
        if flag_nouns == 0:
            out_nouns.write(str(-1))

        out_noun.write('\n')
        out_nouns.write('\n')

for i in noun:
    out_all_noun.write(i+'\n')
for i in nouns:
    out_all_nouns.write(i+'\n')
     
nlp.close()
out_all_noun.close() 
out_all_nouns.close() 

count = 0
# color & people
with open(path_multi30k, 'r', encoding='utf-8') as f:
    for sentence in tqdm(f):
        sentence = sentence.strip().split()
        count += 1
        flag_color = 0
        flag_people = 0
        for idx, i in enumerate(sentence):
            if i in colors:
                out_color.write(str(idx)+' ')
                flag_color = 1
                num_c += 1
            elif i in people:
                out_people.write(str(idx)+' ')
                flag_people = 1     
                num_p += 1
   
        if flag_color == 0:
            out_color.write(str(-1))
        if flag_people == 0:
            out_people.write(str(-1))

        out_color.write('\n')
        out_people.write('\n')

out_people.close()
out_color.close() 
out_noun.close() 
out_nouns.close()
print(num_p, num_c, num_n, num_ns) 

def masking(x, pos, flag):
    for i in pos:
        if i == '-1':
            return
        else:
            x[int(i)] = flag

# create masking text
in_people = open('data/masking/multi30k.people.position', 'r', encoding='utf-8')
in_color = open('data/masking/multi30k.color.position', 'r', encoding='utf-8')
in_noun = open('data/masking/multi30k.noun.position', 'r', encoding='utf-8')
in_nouns = open('data/masking/multi30k.nouns.position', 'r', encoding='utf-8')
_in = open(path_multi30k, 'r', encoding='utf-8')
for p,c,n,ns,l in zip(in_people, in_color, in_noun, in_nouns, _in):
    x = l.strip().split()
    masking(x, p.strip().split(), '[MASK_P]')
    masking(x, c.strip().split(), '[MASK_C]')
    masking(x, n.strip().split(), '[MASK_N]')
    masking(x, ns.strip().split(), '[MASK_NS]')
    out.write(' '.join(x)+'\n')
    
out.close() 
in_people.close()
in_color.close()
in_noun.close()
in_nouns.close()
_in.close()
