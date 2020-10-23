OS := $(shell uname)
OPTIONS:= 

ifeq ($(OS),Darwin)
	OPTIONS += -framework OpenCL
else
	OPTIONS += -l OpenCL
endif

main: main.cpp
	g++ -O2 -Wno-deprecated-declarations -g main.cpp -o main $(OPTIONS)

clean:
	rm -rf main
