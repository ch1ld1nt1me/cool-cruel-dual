import os.path
from fpylll import IntegerMatrix, LLL
from sage.all import next_prime, RDF, sqrt, GF, ZZ
from sage.all import random_matrix, block_matrix, identity_matrix, zero_matrix, matrix


def normProfile(B, q):
    L = [ RDF( sqrt(12/(q**2-1)) * v.norm() / sqrt( len(v) - 1 ) ) for v in B ] 
    csv = "i,s\n"
    for i, l in enumerate(L):
        csv += "%d,%f\n" % (i,l)
    return csv


def gen_data(n, log_q, fn):
    q = next_prime(2**log_q)
    print("n = %d, q = 2^%d" % (n, log_q))
    A = random_matrix(GF(q), n).change_ring(ZZ)

    B = IntegerMatrix.from_matrix( block_matrix([
        [zero_matrix(n), q*identity_matrix(n)],
        [identity_matrix(n), A]
    ]) )

    LLL.reduction(B)
    B = matrix(B)

    R = B.submatrix(0,0,2*n,n)
    left = B.submatrix(0,n,2*n,n)
    left_t = left.transpose()

    A_red = R*A % q

    for i in range(A_red.nrows()):
        for j in range(A_red.ncols()):
            if A_red[i,j] > q/2:
                A_red[i,j] -= q

    A_red_t = A_red.transpose()

    i = 0
    while B[i][n:].norm() == q:
        i += 1
    q_vectors = i

    csv = normProfile(left_t, q)
    with open(fn, "w") as f:
        f.writelines(csv)

    return q_vectors


def gen_latex(n, log_q_range, q_vectors, fn):
    tex = ""
    tex += "\\documentclass{article}\n"
    tex += "\\usepackage[a4paper, total={6in, 8in}]{geometry}\n"
    tex += "\\usepackage{pgfplots}\n"
    tex += "\\begin{document}\n"
    tex += "\\begin{figure}\n"
    tex += "\\centering\n"

    for i in range(len(log_q_range)):
        log_q = log_q_range[i]
        qs = q_vectors[i]
        if i % 2 == 0 and i > 0:
            tex += "\\\\\\vspace*{0.4cm}\n"
        tex += "\\begin{tikzpicture}\n"
        tex += "\\begin{axis}[\n"
        tex += "%%ymode=log,\n"
        tex += "axis lines=left,\n"
        tex += "cycle list name=exotic,\n"
        tex += "%%width = 1.1\\textwidth, \n"
        tex += "%%height = 0.8\\textwidth,\n"
        tex += "grid = major,\n"
        tex += "grid style = {dashed, gray!30},\n"
        tex += "%title style={at={(0.5,1.1)},anchor=north},\n"
        tex += "title = { $q = 2^{%d}$ },\n" % log_q
        tex += "x=1,\n"
        tex += "xmin=1,\n"
        tex += "xmax=%d,\n" % n
        tex += "y=60,\n"
        tex += "ymin=0,\n"
        tex += "ymax=1.1,\n"
        tex += "xlabel = {Column index},\n"
        tex += "ylabel = {${\\sqrt{12}\\sigma}/{\\sqrt{q^2-1}}$},\n"
        tex += "xticklabel style = {font=\\scriptsize},\n"
        tex += "yticklabel style = {font=\\scriptsize,\n"
        tex += "    /pgf/number format/fixed,\n"
        tex += "    /pgf/number format/precision=1 },\n"
        tex += "]\n"
        tex += "\\addplot table [x=i, y=s, col sep=comma, only marks] {%d.csv};\n" % log_q
        tex += "\\draw [ultra thick, red!60] (%d,0) -- (%d,100);\n" % (qs, qs)
        tex += "\\end{axis}\n"
        tex += "\\end{tikzpicture}\n"

    tex += "\\end{figure}\n"
    tex += "\\end{document}\n"

    with open(fn, "w") as f:
        f.writelines(tex)


if __name__ == "__main__":
    # generate plot data
    n = 128
    log_q_range = [5, 10, 15, 20]
    q_vectors = []
    for log_q in log_q_range:
        fn = os.path.join(".", "cruel_bits", f"{log_q}.csv")
        qs = gen_data(n, log_q, fn)
        q_vectors.append(qs)

    # generate latex harness
    fn = os.path.join(".", "cruel_bits", "figure.tex")
    gen_latex(n, log_q_range, q_vectors, fn)
