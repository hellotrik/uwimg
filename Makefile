OPENCV=0
OPENMP=1
DEBUG=0
LIB=1

OBJ=image_opencv.o load_image.o args.o filter_image.o test.o harris_image.o matrix.o panorama_image.o flow_image.o list.o data.o net.o activations.o batch_norm.o connected_layer.o convolutional_layer.o maxpool_layer.o classifier.o 
EXOBJ=main.o

VPATH=./mysrc/:./:./mysrc/uwimg:./mysrc/uwnet
SLIB=libuwimg.so
ALIB=libuwimg.a
EXEC=uwtrik
OBJDIR=./obj/

CC=gcc -m64
CPP=g++ -std=c++11 -m64
AR=ar
ARFLAGS=rcs
OPTS=-Ofast
LDFLAGS= -lm -pthread 
COMMON= -Iinclude/ -Isrc/ 
CFLAGS=-Wall -Wno-unknown-pragmas -Wfatal-errors -fPIC

ifeq ($(OPENMP), 1) 
CFLAGS+= -fopenmp
endif

ifeq ($(DEBUG), 1) 
OPTS=-O0 -g
COMMON= -Iinclude/ -Isrc/ 
else
CFLAGS+= -flto
endif

CFLAGS+=$(OPTS)

ifeq ($(OPENCV), 1) 
COMMON+= -DOPENCV
CFLAGS+= -DOPENCV
LDFLAGS+= `pkg-config --libs opencv` -lstdc++  # This may need to be opencv4 or a specific path
COMMON+= `pkg-config --cflags opencv`
endif

EXOBJS = $(addprefix $(OBJDIR), $(EXOBJ))
OBJS = $(addprefix $(OBJDIR), $(OBJ))
DEPS = $(wildcard src/*.h) Makefile 


ifeq ($(LIB), 1) 
all: obj $(SLIB) $(ALIB) $(EXEC)
else
all: obj $(EXEC)
endif

$(EXEC): $(EXOBJS) $(OBJS)
	$(CC) $(COMMON) $(CFLAGS) $^ -o $@ $(LDFLAGS) 

$(ALIB): $(OBJS)
	$(AR) $(ARFLAGS) $@ $^

$(SLIB): $(OBJS)
	$(CC) $(CFLAGS) -shared $^ -o $@ $(LDFLAGS)

$(OBJDIR)%.o: %.c $(DEPS)
	$(CC) $(COMMON) $(CFLAGS) -c $< -o $@

$(OBJDIR)%.o: %.cpp $(DEPS)
	$(CPP) $(COMMON) $(CFLAGS) -c $< -o $@

obj:
	mkdir obj

.PHONY: clean

clean:
	rm -rf $(OBJS) $(SLIB) $(ALIB) $(EXEC) $(EXOBJS) $(OBJDIR)/*

