# ---- Link --------------------------- 
_C_uflow.so:  C_uflow.o Makefile
	g++ -shared C_uflow.o -o _C_uflow.so  

# ---- g++ C compile ------------------
C_uflow.o:  C_uflow.c C_uflow.h Makefile
	g++ -fPIC -c C_uflow.c -I/usr/include/python2.7 -I/usr/include/python2.7/Numeric

clean:
	rm -f *.o *.so core
