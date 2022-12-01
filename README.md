# Learning Debiased Classifier with Biased Committee

## Learning main classifier and auxiliary classifiers
```
cd ..
cd LWBC
```

## Data
please downloads npyfiles from  below link and place the folders under 'npy_files/' 

https://drive.google.com/drive/folders/1D5Ma-r7zVBdWGmugCEwMno1nq_kJyoK7?usp=share_link

```
tar -xvf celebA_Blond_Hair.tar
mv celebA_Blond_Hair npy_files/
tar -xvf BAR.tar
mv BAR npy_files/
tar -xvf NICO.tar
mv NICO npy_files/
tar -xvf imagenet.tar
mv imagenet npy_files/
```

## Training
CelebA, target: Blond_Hair
```
bash run_celebA.sh
```
NICO
```
bash run_NICO.sh
```
BAR
```
bash run_BAR.sh
```
Imagenet9 & Imagenet-A
```
bash run_imagenet.sh
```



# BYOL implementation
Simple BYOL implementation.
```
cd SSL
```

## Data 
CelebA: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html


To run, please see byol/byol_celebA.sh or byoll/byol_imagenet9.sh
```
bash byol_celebA.sh
```

This code is orginated from https://github.com/CupidJay/Scaled-down-self-supervised-learning.
We borrow byol folder from above repository.



## LICENSE
```
MIT License

Copyright (c) 2022 Nayeong-V-Kim

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```