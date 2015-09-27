all:main.cpp sift.cpp
	g++ -std=c++11 -o sift.o -O3 main.cpp sift.cpp `pkg-config opencv --libs --cflags`
