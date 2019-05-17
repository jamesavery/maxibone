# Input:
#  M: (num_matrices,3,3) a list of symmetric real matrices
#
float_ensured_digits = {'float32':7,'float64':16,'float96':16};
def eigenvalues3x3(M):
    M       = uk.make_behaving(M);
    lambdas = bh.empty(len(M),dtype=M.dtype);
    kernel = read_kernel("matrix3x3/eigenvalues") % {'real_t':ctype[M.dtype.name],
                                                     'float_digits':float_ensured_digits[M.dtype.name],
                                                     'num_matrices':len(M)};
    uk.execute(kernel,[lambdas,M]);
