Fixed **Open in notebook** giving a notebook that failed on its first line when the
derivation used imports, constants or helpers from elsewhere in its file or project.
A cell ending on a derivation now shows the result it returns in production and can be
run again, and running one cell first runs the cells it depends on. Fixed dollar
amounts in notebook markdown rendering as math.
