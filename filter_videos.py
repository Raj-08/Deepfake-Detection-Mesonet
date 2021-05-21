import json
import os
import shutil
Dir='' ## location of DFDC dataset folder
Dir_Fake= '' ## location of Folder where you want fake videos
Dir_Real= '' ## location of Folder where you want real videos

alls1=os.listdir(Dir)
for a in alls1:
    alls2=os.listdir(Dir+a)
    alls2=[v for v in alls2 if '.json' in v]
    for i in alls2:    
        with open(Dir+a+'/'+i) as f:
            print(Dir+a+'/'+i)
            data = json.load(f)
            for i in data:
                try:
                    if data[i]['label'] == 'FAKE':
                        print(data[i])
                        shutil.copy(Dir+a+'/'+i,Dir_Fake+i)
                        original=data[i]['original']
                        shutil.copy(Dir+a+'/'+original,Dir_Real+original)
                    elif data[i]['label'] == 'REAL':
                        shutil.copy(Dir+a+'/'+i,Dir_Real+i)
                        print(data[i])
                except:
                    print('missed')
