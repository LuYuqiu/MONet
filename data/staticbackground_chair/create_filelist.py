import os

finddir='/home/user/Luyuqiu/flownet2/data/staticbackground_chair/staticbackground_TEST/'
dir='staticbackground_chair/staticbackground_TEST/'

f=open('/home/user/Luyuqiu/flownet2/data/staticbackground_chair/test_list.txt',"w")

for i in range(14001):
    findimg = os.path.join(finddir+str(i)+'_gt.png')
    if os.path.exists(findimg):
        img1 = os.path.join(dir+str(i)+'_img1.png')
        img2 = os.path.join(dir+str(i)+'_img2.png')
        gt = os.path.join(dir+str(i)+'_gt.png')
        line = ' '.join([img1,img2,gt])
        f.write(line)
        f.write('\n')
f.close()
    



