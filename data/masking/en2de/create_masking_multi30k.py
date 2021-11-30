import os
import shutil

data_path = os.path.abspath(os.path.join(os.getcwd(), "../.."))

dic1 = {
    'maskc': 'color',
    'maskp': 'people',
    'maskn': 'noun',
    'maskns': 'nouns',
}

dic2 = {
    'maskc': '[MASK_C]',
    'maskp': '[MASK_P]',
    'maskn': '[MASK_N]',
    'maskns': '[MASK_NS]',
}

train_lines = 29000
val_lines = 1014
test_2016_lines = 1000
test_2017_lines = 1000
test_coco_lines = 461

if __name__ == "__main__":
    language = 'multi30k-en-de'
    for mask_token in ['maskc', 'maskp']:
        new_dir = os.path.join(data_path, language+'.'+mask_token)

        if not os.path.exists(new_dir):
            os.mkdir(new_dir)
        
        pos = open('multi30k.'+dic1[mask_token]+'.bpe.position', 'r', encoding='utf-8')
        multi30k = open(os.path.join(data_path, 'multi30k', language+'.bpe.en'), 'r', encoding='utf-8')
        out_train = open(os.path.join(new_dir, 'train.en'), 'w', encoding='utf-8')
        out_valid = open(os.path.join(new_dir, 'valid.en'), 'w', encoding='utf-8')
        out_test_2016 = open(os.path.join(new_dir, 'test.2016.en'), 'w', encoding='utf-8')
        out_test_2017 = open(os.path.join(new_dir, 'test.2017.en'), 'w', encoding='utf-8')
        out_test_coco = open(os.path.join(new_dir, 'test.coco.en'), 'w', encoding='utf-8')

        tmp = []
        for line, position in zip(multi30k, pos):
            x = position.strip().split()
            lines = line.strip().split()
            for i in x:
                if i == '-1':
                    break
                else:
                    lines[int(i)] = dic2[mask_token]
            
            tmp.append(' '.join(lines)+'\n')

        # write
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
        
        pos.close()
        multi30k.close()
        out_train.close()
        out_valid.close()
        out_test_2016.close()
        out_test_2017.close()
        out_test_coco.close()

        # copy target language file
        target_language = language.split('-')[-1]
        shutil.copyfile(os.path.join(data_path, language, 'train.'+target_language), os.path.join(new_dir, 'train.'+target_language))
        shutil.copyfile(os.path.join(data_path, language, 'valid.'+target_language), os.path.join(new_dir, 'valid.'+target_language))
        shutil.copyfile(os.path.join(data_path, language, 'test.2016.'+target_language), os.path.join(new_dir, 'test.2016.'+target_language))
        shutil.copyfile(os.path.join(data_path, language, 'test.2017.'+target_language), os.path.join(new_dir, 'test.2017.'+target_language))
        shutil.copyfile(os.path.join(data_path, language, 'test.coco.'+target_language), os.path.join(new_dir, 'test.coco.'+target_language))