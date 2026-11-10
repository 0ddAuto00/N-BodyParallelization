CC      ?= cc
TARGET  := nbody
SRCS    := nbody.c
OBJS    := $(SRCS:.c=.o)

CFLAGS  := -std=c11 -Wall -Wextra -Wpedantic -O2
LDLIBS  := -lm

# Pass extra flags on the command line, e.g. `make CFLAGS_EXTRA="-march=native -ffast-math"`
CFLAGS  += $(CFLAGS_EXTRA)

.PHONY: all debug run clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(LDLIBS)

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

# Debug build with sanitizers
debug: CFLAGS := -std=c11 -Wall -Wextra -Wpedantic -O0 -g -fsanitize=address,undefined
debug: LDFLAGS += -fsanitize=address,undefined
debug: clean $(TARGET)

run: $(TARGET)
	./$(TARGET) nbody.cfg

clean:
	rm -f $(TARGET) $(OBJS)
