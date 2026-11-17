CC      ?= cc
TARGET  := nbody
SRCDIR  := source
OBJDIR  := object
SRCS    := $(wildcard $(SRCDIR)/*.c)
OBJS    := $(SRCS:$(SRCDIR)/%.c=$(OBJDIR)/%.o)

CFLAGS  := -std=c11 -Wall -Wextra -Wpedantic -O2 -fopenmp -I$(SRCDIR)
LDLIBS  := -lm
LDFLAGS := -fopenmp

# Pass extra flags on the command line, e.g. `make CFLAGS_EXTRA="-march=native -ffast-math"`
CFLAGS  += $(CFLAGS_EXTRA)

.PHONY: all debug run clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(LDLIBS)

$(OBJDIR)/%.o: $(SRCDIR)/%.c | $(OBJDIR)
	$(CC) $(CFLAGS) -c -o $@ $<

$(OBJDIR):
	mkdir -p $@

# Debug build with sanitizers
debug: CFLAGS := -std=c11 -Wall -Wextra -Wpedantic -O0 -g -fopenmp -fsanitize=address,undefined -I$(SRCDIR)
debug: LDFLAGS += -fsanitize=address,undefined
debug: clean $(TARGET)

run: $(TARGET)
	./$(TARGET) nbody.cfg

clean:
	rm -rf $(OBJDIR) $(TARGET)