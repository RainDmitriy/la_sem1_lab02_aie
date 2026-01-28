from base import Matrix
from type import CSCData, CSCIndices, CSCIndptr, Shape, DenseMatrix, COOData, COORows, COOCols
from COO import COOMatrix
from collections import defaultdict

class CSCMatrix(Matrix):
    def __init__(self, data: CSCData, indices: CSCIndices, indptr: CSCIndptr, shape: Shape):
        super().__init__(shape)
        self.data = data
        self.indices = indices
        self.indptr = indptr
        if len(indptr) != shape[1] + 1:
            raise ValueError(f"indptr должен иметь длину shape[1] + 1 = {shape[1] + 1}")

    def to_dense(self) -> DenseMatrix:
        rows, cols = self.shape
        dense = [[0.0] * cols for _ in range(rows)]

        for j in range(cols):
            start, end = self.indptr[j], self.indptr[j + 1]
            for pos in range(start, end):
                i = self.indices[pos]
                val = self.data[pos]
                dense[i][j] = val

        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        if not isinstance(other, CSCMatrix):
            other_csc = other._to_csc() if hasattr(other, '_to_csc') else CSCMatrix.from_dense(other.to_dense())
        else:
            other_csc = other

        rows, cols = self.shape
        result_data = []
        result_indices = []
        result_indptr = [0]

        for j in range(cols):
            col_dict = defaultdict(float)

            start1, end1 = self.indptr[j], self.indptr[j + 1]
            for pos in range(start1, end1):
                row = self.indices[pos]
                col_dict[row] += self.data[pos]

            start2, end2 = other_csc.indptr[j], other_csc.indptr[j + 1]
            for pos in range(start2, end2):
                row = other_csc.indices[pos]
                col_dict[row] += other_csc.data[pos]

            sorted_rows = sorted(col_dict.keys())
            for row in sorted_rows:
                val = col_dict[row]
                if abs(val) > 1e-12:
                    result_data.append(val)
                    result_indices.append(row)

            result_indptr.append(len(result_data))

        return CSCMatrix(result_data, result_indices, result_indptr, self.shape)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        new_data = [val * scalar for val in self.data]
        return CSCMatrix(new_data, self.indices.copy(), self.indptr.copy(), self.shape)

    def transpose(self) -> 'Matrix':
        from CSR import CSRMatrix
        coo = self._to_coo()
        transposed_coo = coo.transpose()
        return transposed_coo._to_csr()

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        from CSR import CSRMatrix
        from COO import COOMatrix

        csr_self = self.transpose().transpose()

        if isinstance(other, CSCMatrix):
            csr_other = other.transpose().transpose()
            result_csr = csr_self._matmul_impl(csr_other)
            return result_csr._to_csc()
        else:
            return csr_self._matmul_impl(other)

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSCMatrix':
        rows = len(dense_matrix)
        cols = len(dense_matrix[0]) if rows > 0 else 0

        coo = COOMatrix.from_dense(dense_matrix)
        return coo._to_csc()

    def _to_csr(self) -> 'CSRMatrix':
        return self.transpose()

    def _to_coo(self) -> 'COOMatrix':
        rows, cols = self.shape
        data = []
        row_indices = []
        col_indices = []

        for j in range(cols):
            start, end = self.indptr[j], self.indptr[j + 1]
            for pos in range(start, end):
                data.append(self.data[pos])
                row_indices.append(self.indices[pos])
                col_indices.append(j)

        indices = sorted(range(len(data)), key=lambda i: (row_indices[i], col_indices[i]))

        sorted_data = [data[i] for i in indices]
        sorted_rows = [row_indices[i] for i in indices]
        sorted_cols = [col_indices[i] for i in indices]

        return COOMatrix(sorted_data, sorted_rows, sorted_cols, self.shape)

    @classmethod
    def from_coo(cls, data: COOData, rows: COORows, cols: COOCols, shape: Shape) -> 'CSCMatrix':
        coo = COOMatrix(data, rows, cols, shape)
        return coo._to_csc()