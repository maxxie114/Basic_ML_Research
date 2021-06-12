EXEC=perceptron
CFLAGS=-Wall -Wextra -Werror -Wpedantic -std=c99

all: perceptron.o
	clang -o $(EXEC) $^ $(CFLAGS) -lm

debug: perceptron.o
	clang -g -o $(EXEC) $^ $(CFLAGS) -lm

clean:
	rm -rf $(EXEC) *.o

valgrind:
	make clean; make debug;
	valgrind --leak-check=full ./$(EXEC)
