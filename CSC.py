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
        if abs(scalar) < 1e-14:
            return CSCMatrix([], [], [0] * (self.shape[1] + 1), self.shape)
        new_data = [val * scalar for val in self.data]
        return CSCMatrix(new_data, self.indices.copy(), self.indptr.copy(), self.shape)

    def transpose(self) -> 'Matrix':
        from CSR import CSRMatrix

        rows, cols = self.shape
        nnz = len(self.data)

        if nnz == 0:
            return CSRMatrix([], [], [0] * (rows + 1), (cols, rows))

        row_counts = [0] * rows
        for j in range(cols):
            start, end = self.indptr[j], self.indptr[j + 1]
            for pos in range(start, end):
                i = self.indices[pos]
                row_counts[i] += 1

        csr_indptr = [0] * (rows + 1)
        for i in range(rows):
            csr_indptr[i + 1] = csr_indptr[i] + row_counts[i]

        csr_data = [0.0] * nnz
        csr_indices = [0] * nnz

        current_pos = csr_indptr.copy()

        for j in range(cols):
            start, end = self.indptr[j], self.indptr[j + 1]
            for pos in range(start, end):
                i = self.indices[pos]
                val = self.data[pos]

                pos_in_csr = current_pos[i]
                csr_data[pos_in_csr] = val
                csr_indices[pos_in_csr] = j
                current_pos[i] += 1

        return CSRMatrix(csr_data, csr_indices, csr_indptr, (cols, rows))

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        from CSR import CSRMatrix

        csr_self = self.transpose()
        if isinstance(other, CSCMatrix):
            csr_other = other.transpose()
            result = csr_self._matmul_impl(csr_other)
            return result.transpose()
        else:
            return csr_self._matmul_impl(other).transpose()

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSCMatrix':
        rows = len(dense_matrix)
        cols = len(dense_matrix[0]) if rows > 0 else 0

        data = []
        indices = []
        col_indices = []

        for j in range(cols):
            for i in range(rows):
                val = dense_matrix[i][j]
                if abs(val) > 1e-12:
                    data.append(val)
                    indices.append(i)
                    col_indices.append(j)

        if not data:
            return cls([], [], [0] * (cols + 1), (rows, cols))

        sorted_indices = sorted(range(len(data)), key=lambda idx: (col_indices[idx], indices[idx]))

        sorted_data = [data[idx] for idx in sorted_indices]
        sorted_indices_list = [indices[idx] for idx in sorted_indices]
        sorted_col_indices = [col_indices[idx] for idx in sorted_indices]

        col_counts = [0] * cols
        for j in sorted_col_indices:
            col_counts[j] += 1

        indptr = [0] * (cols + 1)
        for j in range(cols):
            indptr[j + 1] = indptr[j] + col_counts[j]

        return cls(sorted_data, sorted_indices_list, indptr, (rows, cols))

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

        return COOMatrix(data, row_indices, col_indices, self.shape)

    @classmethod
    def from_coo(cls, data: COOData, rows: COORows, cols: COOCols, shape: Shape) -> 'CSCMatrix':
        coo = COOMatrix(data, rows, cols, shape)
        return coo._to_csc()