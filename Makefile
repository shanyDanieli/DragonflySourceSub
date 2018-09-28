all:

clean:

install: $(PROGRAMS)
	cp sourcesub.py /usr/local/bin/sourcesub; chmod a+x /usr/local/bin/sourcesub
