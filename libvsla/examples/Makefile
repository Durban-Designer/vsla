# Makefile for VSLA examples

CC = gcc
CFLAGS = -std=c99 -Wall -Wextra -I../include
LIBS = -lm
LIBVSLA = ../build/libvsla.a

# Source files
SOURCES = basic_usage.c
EXAMPLES = $(SOURCES:.c=)

.PHONY: all clean

all: $(EXAMPLES)

%: %.c $(LIBVSLA)
	$(CC) $(CFLAGS) $< $(LIBVSLA) $(LIBS) -o $@

$(LIBVSLA):
	@echo "Building library..."
	@cd .. && mkdir -p build && cd build && cmake .. && make

clean:
	rm -f $(EXAMPLES)

run-basic: basic_usage
	./basic_usage

help:
	@echo "Available targets:"
	@echo "  all        - Build all examples"
	@echo "  basic_usage - Build basic usage example"
	@echo "  run-basic  - Build and run basic usage example"
	@echo "  clean      - Remove built examples"