copy code and run on ug machines (e.g. ug66)
the code on GPU is very slow (~30s for each epoch)
you can change epoch value in function training in training_kernel.cl

to run:
typer "make"
typer "time ./main"