# extract image feature via DETR
import torch
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm
import os
import argparse

# running following code, to download DETR offical code and model
# see https://github.com/facebookresearch/detr
model = torch.hub.load('facebookresearch/detr', 'detr_resnet101_dc5', pretrained=True).to('cuda:0')
model.eval()

def get_filenames(path):
    l = []
    with open(path, 'r') as f:
        for line in f:
            l.append(line.strip().split('#')[0])
    return l

dic = {
    'test2017': 'test2017', 
    'testcoco': 'testcoco',
    'test2016': 'flickr30k',
    'train': 'flickr30k',
    'val': 'flickr30k',
    }

dic1 = {
    'test2017': 'test_2017_flickr.txt',
    'testcoco': 'test_2017_mscoco.txt',
    'test2016': 'test_2016_flickr.txt',
    'train': 'train.txt',
    'val': 'val.txt',
    }

dic2 = {
    'test2017': 'test1', 
    'testcoco': 'test2',
    'test2016': 'test',
    'train': 'train',
    'val': 'valid',
    }

transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='which dataset')
    parser.add_argument('--dataset', type=str, choices=['train', 'val', 'test2016', 'test2017', 'testcoco'], help='which dataset')
    parser.add_argument('--path', type=str)
    args = parser.parse_args()

    flickr30k_path = args.path
    dataset = args.dataset
    save_dir = os.path.join('data', 'DETR-DC5-R101')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print('extracting ' + dataset + '\'s image feature from DETR-DC5-R101') 
    
    count = 1
    filenames = get_filenames(os.path.join(flickr30k_path, dic1[dataset]))
    tmp = []

    # propagate through the model
    with torch.no_grad():
        for i in tqdm(filenames):
            i = os.path.join(flickr30k_path, dic[dataset]+'-images', i)
            
            img = Image.open(i)
            # mean-std normalize the input image (batch-size: 1)
            input = transform(img).unsqueeze(0).to('cuda:0') # transform and add batch dimension
            # outputs return 6 decoder layers' features
            # we get the lastest layer's feature
            out = model(input)[-1]
            
            tmp.append(out.detach().to('cuda:1'))
            if len(tmp) == 2000:
                res = torch.cat(tmp).cpu()
                print(res.shape)
                torch.save(res, os.path.join(save_dir, str(count)+dic2[dataset]+'.pth'))
                count += 1
                tmp = []
    
        res = torch.cat(tmp).cpu()
        if count > 1:
            torch.save(res, os.path.join(save_dir, 'final'+dic2[dataset]+'.pth'))
        else:
            print('feature shape:', res.shape, ',save in:', save_dir+'/'+dic2[dataset]+'.pth')
            torch.save(res, os.path.join(save_dir, dic2[dataset]+'.pth'))
        del tmp
    
        _tmp = []
        if count > 1:
            for i in range(1, count):
                _tmp.append(torch.load(os.path.join(save_dir, str(i)+dic2[dataset]+'.pth')))
            _tmp.append(torch.load(os.path.join(save_dir, 'final'+dic2[dataset]+'.pth')))
            res = torch.cat(_tmp).cpu()
            print('feature shape:', res.shape, ',save in:', save_dir+'/'+dic2[dataset]+'.pth')
            torch.save(res, os.path.join(save_dir, dic2[dataset]+'.pth'))
            
            # delete  
            for i in range(1, count):
                os.remove(os.path.join(save_dir, str(i)+dic2[dataset]+'.pth'))
            os.remove(os.path.join(save_dir, 'final'+dic2[dataset]+'.pth'))
