from flickr30k_entities.flickr30k_entities_utils import *
from collections import defaultdict
from stanfordcorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP(r'../stanford-corenlp-4.3.2')#, lang='de')

def get_sentence_list():
    sentence_list = []
    with open('train_val_test2016.en','r') as f:
        for l in f:
            sentence_list.append(l.strip())
    return sentence_list

def filter_EscapeString(l):
    l = l.replace('&apos;', '\'')
    l = l.replace("&amp;", "&")
    l = l.replace("& amp ;", '&')
    l = l.replace("&quot;", '"')
    return l

def get_name_list():
    name_list = []
    with open('train_val_test2016.txt','r') as f:
        for i in f:
            name_list.append(i.split('.')[0])   
    return name_list

def fix_post_tag(phrase_pos_tag, phrase):
    tmp = []
    tmp_idx = 0
    words = phrase.split()
    for idx, i in enumerate(words):
        if i == phrase_pos_tag[tmp_idx][0]:
            tmp.append(phrase_pos_tag[tmp_idx])
            tmp_idx += 1
        else:
            str1 = phrase_pos_tag[tmp_idx][0]
            tmp_idx += 1
            while str1 != i:
                str1 += phrase_pos_tag[tmp_idx][0]
                tmp_idx += 1
            tmp.append((i, 'UNK'))
    return tmp

def write_dict(filename, dic):
    out = open(filename, 'w', encoding='utf-8')
    t = sorted(dic.items(), key=lambda item:item[1])
    for i in t:
        out.write(i[0] + ' ' + i[1])
    out.close()

if __name__ == "__main__":
    noun = defaultdict(int)
    nouns = defaultdict(int)
    #people = defaultdict(int)
    name_list = get_name_list()
    sentence_list = get_sentence_list()
 
    for index in range(len(name_list)):
        image = name_list[index]
        origin_sentence = sentence_list[index]
        sentence = filter_EscapeString(origin_sentence)

        # a list
        x = get_sentence_data('flickr30k_entities/Sentences/'+image.split('.')[0]+'.txt')

        for j in x:
            entity_sentence = j['sentence'].replace(' ','').replace('”','"').replace('`','\'').replace('"','').lower()
            if entity_sentence == sentence.replace('"','').replace(' ',''):
                for t in j['phrases']:
                    phrase = t['phrase'].lower()
                    # if 'people' in t['phrase_type']:
                    try:
                        phrase_pos_tag = nlp.pos_tag(phrase)
                        if len(phrase_pos_tag) > len(phrase.split()):
                            fix_post_tag(phrase_pos_tag, phrase)
                        assert len(phrase_pos_tag) == len(phrase.split()):
                    except:
                        print(phrase)

                    #for pos_tag in phrase_pos_tag:
                    #   if pos_tag[1] == 'NN':
                    #       noun[pos_tag[0]] += 1
                    #   elif pos_tag[1] == 'NNS':
                    #       nouns[pos_tag[0]] += 1
                break

    write_dict('data/masking/noun.en', noun)
    write_dict('data/masking/nouns.en', nouns)
