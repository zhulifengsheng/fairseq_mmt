import random
import os
import shutil

data_path = os.path.abspath(os.path.join(os.getcwd(), "../.."))

train_lines = 29000
val_lines = 1014
test_2016_lines = 1000
test_2017_lines = 1000
test_coco_lines = 461

dic = {
    'c': '[MASK_C]',
    'p': '[MASK_P]',
    'n': '[MASK_N]',
    'ns': '[MASK_NS]',
}

pos_color = open('../multi30k.color.position', 'r', encoding='utf-8')
pos_noun = open('../multi30k.noun.position', 'r', encoding='utf-8')
pos_nouns = open('../multi30k.nouns.position', 'r', encoding='utf-8')
pos_people = open('../multi30k.people.position', 'r', encoding='utf-8')

#pos_bpe_color = open('multi30k.color.bpe.position', 'w', encoding='utf-8')
#pos_bpe_noun = open('multi30k.noun.bpe.position', 'w', encoding='utf-8')
#pos_bpe_nouns = open('multi30k.nouns.bpe.position', 'w', encoding='utf-8')
#pos_bpe_people = open('multi30k.people.bpe.position', 'w', encoding='utf-8')

def record_origin_pos(pos):
    l = []
    for line in pos:
        l.append(line.strip().split())
    return l

def get_position(f, o, l):
    num = 0
    for line in f:
        if line[0] != '-1':
            for i in line:
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

def get_matching():
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
    return _l

def get_mask_bpe_pos(l, list_matching):
    new_l = []
    for i, j in zip(l, list_matching):
        #print(i ,j)
        tmp = []
        for tuple_pos in i:
            if tuple_pos[0] in j.keys():
                for bpe_pos in j[tuple_pos[0]]:
                    tmp.append((bpe_pos, tuple_pos[1]))
            else:
                tmp.append(tuple_pos)
        new_l.append(tmp)
    return new_l

if __name__ == '__main__':
    list_matching = get_matching()
    
    # record origin text data's position
    _pos_people = record_origin_pos(pos_people)
    _pos_color = record_origin_pos(pos_color)
    _pos_noun = record_origin_pos(pos_noun)
    _pos_nouns = record_origin_pos(pos_nouns)
    pos_people.close()    
    pos_color.close()
    pos_noun.close()
    pos_nouns.close()

    #get_position(_pos_color, pos_bpe_color, list_matching)
    #get_position(_pos_people, pos_bpe_people, list_matching)
    #get_position(_pos_noun, pos_bpe_noun, list_matching)    
    #get_position(_pos_nouns, pos_bpe_nouns, list_matching)
    #pos_bpe_people.close()    
    #pos_bpe_color.close()
    #pos_bpe_noun.close()
    #pos_bpe_nouns.close()

    # masking 1-4
    for num in range(1, 5):
        l = []  # list of masking origin text data's position
        for p, c, n, ns in zip(_pos_people, _pos_color, _pos_noun, _pos_nouns):
            where = []
            #if p[0] != '-1':
            #    for i in p:
            #        where.append((i, 'p'))
            #if c[0] != '-1':
            #    for i in c:
            #        where.append((i, 'c'))
            if n[0] != '-1':
                for i in n:
                    where.append((i, 'n'))
            if ns[0] != '-1':
                for i in ns:
                    where.append((i, 'ns'))
            
            if len(where) > num:
                where = random.sample(where, num)
            l.append(where)

        language = 'multi30k-en-de'
        mask_token = 'mask'+str(num)
        new_dir = os.path.join(data_path, language+'.'+mask_token)

        if not os.path.exists(new_dir):
            os.mkdir(new_dir)

        multi30k = open(os.path.join(data_path, 'multi30k', language+'.bpe.en'), 'r', encoding='utf-8')
        out_train = open(os.path.join(new_dir, 'train.en'), 'w', encoding='utf-8')
        out_valid = open(os.path.join(new_dir, 'valid.en'), 'w', encoding='utf-8')
        out_test_2016 = open(os.path.join(new_dir, 'test.2016.en'), 'w', encoding='utf-8')
        out_test_2017 = open(os.path.join(new_dir, 'test.2017.en'), 'w', encoding='utf-8')
        out_test_coco = open(os.path.join(new_dir, 'test.coco.en'), 'w', encoding='utf-8')

        new_l = get_mask_bpe_pos(l, list_matching)

        # write
        tmp = []
        for line, position in zip(multi30k, new_l):
            lines = line.strip().split()
            for i in position:
                #print(i)
                lines[int(i[0])] = dic[i[1]]
            
            tmp.append(' '.join(lines)+'\n')
            
        for idx, i in enumerate(tmp):
            if idx < train_lines:
                out_train.write(i)
            elif train_lines <= idx and idx < train_lines+val_lines:
                out_valid.write(i)
            elif train_lines+val_lines <= idx and idx < train_lines+val_lines+test_2016_lines:
                out_test_2016.write(i)
            elif train_lines+val_lines+test_2016_lines <= idx and idx < train_lines+val_lines+test_2016_lines+test_2017_lines:
                out_test_2017.write(i)
            else:
                out_test_coco.write(i)

        # copy target language file
        target_language = language.split('-')[-1]
        shutil.copyfile(os.path.join(data_path, language, 'train.'+target_language), os.path.join(new_dir, 'train.'+target_language))
        shutil.copyfile(os.path.join(data_path, language, 'valid.'+target_language), os.path.join(new_dir, 'valid.'+target_language))
        shutil.copyfile(os.path.join(data_path, language, 'test.2016.'+target_language), os.path.join(new_dir, 'test.2016.'+target_language))
        shutil.copyfile(os.path.join(data_path, language, 'test.2017.'+target_language), os.path.join(new_dir, 'test.2017.'+target_language))
        shutil.copyfile(os.path.join(data_path, language, 'test.coco.'+target_language), os.path.join(new_dir, 'test.coco.'+target_language))

        multi30k.close()
        out_train.close()
        out_valid.close()
        out_test_2016.close()
        out_test_2017.close()
        out_test_coco.close()
