CFLAGS ?= -std=c11 -Wall -Wextra -O2
LDFLAGS ?= -lm

beats: beats.c
	$(CC) $(CFLAGS) -o $@ beats.c $(LDFLAGS)

clean:
	rm -f beats
