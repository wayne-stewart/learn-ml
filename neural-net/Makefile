.DEFAULT_GOAL := build

OUTROOT = build
OUTNAME = main
CFLAGS = -Werror -Wall -Wextra -Wpedantic -Wfatal-errors

build: make_dir
	g++ src/main.cpp -o $(OUTROOT)/$(OUTNAME) $(CFLAGS) -O3

run: build
	$(OUTROOT)/$(OUTNAME)

debug: debug-build
	gdb $(OUTROOT)/$(OUTNAME)-debug

debug-build: make_dir
	g++ src/main.cpp -o $(OUTROOT)/$(OUTNAME)-debug $(CFLAGS) -Og -g

release-build: make_dir
	g++ src/main.cpp -o $(OUTROOT)/$(OUTNAME)-release $(CFLAGS) -O3 -DNDEBUG

make_dir:
	mkdir -p $(OUTROOT)


