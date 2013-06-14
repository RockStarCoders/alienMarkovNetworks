# ---- Link --------------------------- 
_C_uflow.so:  C_uflow.o Makefile
	gcc -shared C_uflow.o -o _C_uflow.so  

# ---- gcc C compile ------------------
C_uflow.o:  C_uflow.c C_uflow.h Makefile
	gcc -fPIC -c C_uflow.c -I/usr/include/python2.7 -I/usr/include/python2.7/Numeric

clean:
	rm *.o *.so
