## Count Dice Dots!
Simple program to detect the number of dots on a dice, if present in an image.\
Just place input images in the input images folder. \
Output is saved automatically in the output images folder. 

Author: Hariharan Ramshankar
# How to Run:
This program requires OpenCV 3.x and Cmake. \
Tested on Ubuntu 16.04 LTS \
1)Clone the repo 
```bash
git clone https://github.com/hari0920/CountDiceDots.git
cd CountDiceDots
```
2)Create a build directory 
```bash
mkdir build
cd build
```
3)From the build folder run,
```bash
make && ./CountDiceDots
```

The program prints the total number of images to process, the number of detected dice and the total number of dots.\
Please look at the comments for more debug statements you can use.