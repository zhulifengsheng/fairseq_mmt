t = open('../train_val_test2016.en', 'w', encoding='utf-8')
with open('multi30k/train_val_test2016.en', 'r', encoding='utf-8') as f:
    for l in f:
        l = l.replace('&apos;', '\'')
        l = l.replace("&amp;", "&")
        l = l.replace("& amp ;", '&')
        l = l.replace("&quot;", '"')
        t.write(l)
