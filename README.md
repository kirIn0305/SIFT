# SIFT (Scale-Invariant Feature Transform)

SIFTの勉強!

## Description
SIFTの勉強のために、実装してみました。
Referencesに乗せているSIFTを一から組むブロクのコードのパクリです。
## Requirement
- g++
- OpenCV 2.4.12

## Usage

```bash:terminal
$ g++ -std=c++11 -o sift.o -O3 main.cpp sift.cpp `pkg-config opencv --libs --cflags`
$ ./sift.o
```

## References
>[Gradient ベースの特徴抽出 -SIFTとHOG-](http://www.hci.iis.u-tokyo.ac.jp/~ysato/class14/supplements/sift_tutorial-Fujiyoshi.pdf)  
>[SIFTを一から組むブログ](http://d.hatena.ne.jp/colorcle/20090629/1246292723)
