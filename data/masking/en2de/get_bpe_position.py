pos_color = open('multi30k.color.position', 'r', encoding='utf-8')
pos_noun = open('multi30k.noun.position', 'r', encoding='utf-8')
pos_nouns = open('multi30k.nouns.position', 'r', encoding='utf-8')
pos_people = open('multi30k.people.position', 'r', encoding='utf-8')

pos_bpe_color = open('multi30k.color.bpe.position', 'w', encoding='utf-8')
pos_bpe_noun = open('multi30k.noun.bpe.position', 'w', encoding='utf-8')
pos_bpe_nouns = open('multi30k.nouns.bpe.position', 'w', encoding='utf-8')
pos_bpe_people = open('multi30k.people.bpe.position', 'w', encoding='utf-8')

def get_position(f, o, l):
    num = 0
    for line in f:
        if line.strip() != '-1':
            x = line.strip().split()
            for i in x:
                if i in l[num].keys():  # bpe used on this sentence                      
                    y = l[num][i]
                    for j in y:
                        o.write(j+' ')
                else:
                    o.write(i+' ')
        else:   # no masking token in this line
            o.write('-1')
        o.write('\n')
        num += 1

_l = [] # list of origin2bpe matching
with open('origin2bpe.en-de.match', 'r', encoding='utf-8') as f:
    for sentence in f:
        dic = {}
        if sentence.strip() == '-1':
            _l.append(dic)  # empty dict
        else:
            x = sentence.strip().split(' ')
            for i in x:
                dic[i.split(':')[0]] = i.split(':')[1].split('-')
            _l.append(dic)

get_position(pos_color, pos_bpe_color, _l)
get_position(pos_people, pos_bpe_people, _l)
get_position(pos_noun, pos_bpe_noun, _l)    
get_position(pos_nouns, pos_bpe_nouns, _l)

pos_people.close()    
pos_color.close()
pos_noun.close()
pos_nouns.close()
pos_bpe_people.close()    
pos_bpe_color.close()
pos_bpe_noun.close()
pos_bpe_nouns.close()

# masking 1-4
num = 1
