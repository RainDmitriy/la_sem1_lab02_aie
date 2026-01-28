from base import Matrix
from type import CSRData, CSRIndices, CSRIndptr, Shape, DenseMatrix, COOData, COORows, COOCols
from COO import COOMatrix
from collections import defaultdict


class CSRMatrix(Matrix):
    def __init__(self, data: CSRData, indices: CSRIndices, indptr: CSRIndptr, shape: Shape):
        super().__init__(shape)
        self.data = data
        self.indices = indices
        self.indptr = indptr

        if len(indptr) != shape[0] + 1:
            raise ValueError(f"indptr должен иметь длину shape[0] + 1 = {shape[0] + 1}")

    def to_dense(self) -> DenseMatrix:
        rows, cols = self.shape
        dense = [[0.0] * cols for _ in range(rows)]

        for i in range(rows):
            start, end = self.indptr[i], self.indptr[i + 1]
            for pos in range(start, end):
                j = self.indices[pos]
                val = self.data[pos]
                dense[i][j] = val

        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        if not isinstance(other, CSRMatrix):
            other_csr = other._to_csr() if hasattr(other, '_to_csr') else CSRMatrix.from_dense(other.to_dense())
        else:
            other_csr = other

        rows, cols = self.shape
        result_data = []
        result_indices = []
        result_indptr = [0]

        for i in range(rows):
            row_dict = defaultdict(float)

            start1, end1 = self.indptr[i], self.indptr[i + 1]
            for pos in range(start1, end1):
                col = self.indices[pos]
                row_dict[col] += self.data[pos]

            start2, end2 = other_csr.indptr[i], other_csr.indptr[i + 1]
            for pos in range(start2, end2):
                col = other_csr.indices[pos]
                row_dict[col] += other_csr.data[pos]

            sorted_cols = sorted(row_dict.keys())
            for col in sorted_cols:
                val = row_dict[col]
                if abs(val) > 1e-12:
                    result_data.append(val)
                    result_indices.append(col)

            result_indptr.append(len(result_data))

        return CSRMatrix(result_data, result_indices, result_indptr, self.shape)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        new_data = [val * scalar for val in self.data]
        return CSRMatrix(new_data, self.indices.copy(), self.indptr.copy(), self.shape)

    def transpose(self) -> 'Matrix':
        from CSC import CSCMatrix

        rows, cols = self.shape
        nnz = len(self.data)

        if nnz == 0:
            return CSCMatrix([], [], [0] * (cols + 1), (cols, rows))

        col_counts = [0] * cols
        for i in range(rows):
            start, end = self.indptr[i], self.indptr[i + 1]
            for pos in range(start, end):
                j = self.indices[pos]
                col_counts[j] += 1

        csc_indptr = [0] * (cols + 1)
        for j in range(cols):
            csc_indptr[j + 1] = csc_indptr[j] + col_counts[j]

        csc_data = [0.0] * nnz
        csc_indices = [0] * nnz

        current_pos = csc_indptr.copy()

        for i in range(rows):
            start, end = self.indptr[i], self.indptr[i + 1]
            for pos in range(start, end):
                j = self.indices[pos]
                val = self.data[pos]

                csc_pos = current_pos[j]
                csc_data[csc_pos] = val
                csc_indices[csc_pos] = i
                current_pos[j] += 1

        return CSCMatrix(csc_data, csc_indices, csc_indptr, (cols, rows))

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        rows_A, cols_A = self.shape

        from CSC import CSCMatrix

        if isinstance(other, CSRMatrix):
            rows_B, cols_B = other.shape
            result_data = []
            result_indices = []
            result_indptr = [0]

            for i in range(rows_A):
                row_result = defaultdict(float)

                start_A, end_A = self.indptr[i], self.indptr[i + 1]

                for pos_A in range(start_A, end_A):
                    col_A = self.indices[pos_A]
                    val_A = self.data[pos_A]

                    start_B, end_B = other.indptr[col_A], other.indptr[col_A + 1]

                    for pos_B in range(start_B, end_B):
                        col_B = other.indices[pos_B]
                        val_B = other.data[pos_B]
                        row_result[col_B] += val_A * val_B

                sorted_cols = sorted(row_result.keys())
                for col in sorted_cols:
                    val = row_result[col]
                    if abs(val) > 1e-12:
                        result_data.append(val)
                        result_indices.append(col)

                result_indptr.append(len(result_data))

            return CSRMatrix(result_data, result_indices, result_indptr, (rows_A, cols_B))

        elif isinstance(other, CSCMatrix):
            rows_B, cols_B = other.shape
            result_data = []
            result_indices = []
            result_indptr = [0]

            for i in range(rows_A):
                row_result = defaultdict(float)

                start_A, end_A = self.indptr[i], self.indptr[i + 1]

                for pos_A in range(start_A, end_A):
                    k = self.indices[pos_A]
                    val_A = self.data[pos_A]

                    start_B, end_B = other.indptr[k], other.indptr[k + 1]

                    for pos_B in range(start_B, end_B):
                        j = other.indices[pos_B]
                        val_B = other.data[pos_B]
                        row_result[j] += val_A * val_B

                sorted_cols = sorted(row_result.keys())
                for col in sorted_cols:
                    val = row_result[col]
                    if abs(val) > 1e-12:
                        result_data.append(val)
                        result_indices.append(col)

                result_indptr.append(len(result_data))

            return CSRMatrix(result_data, result_indices, result_indptr, (rows_A, cols_B))

        else:
            other_csr = other._to_csr() if hasattr(other, '_to_csr') else CSRMatrix.from_dense(other.to_dense())
            return self._matmul_impl(other_csr)

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSRMatrix':
        rows = len(dense_matrix)
        cols = len(dense_matrix[0]) if rows > 0 else 0

        data = []
        indices = []
        indptr = [0]

        for i in range(rows):
            row_nnz = 0
            for j in range(cols):
                val = dense_matrix[i][j]
                if abs(val) > 1e-12:
                    data.append(val)
                    indices.append(j)
                    row_nnz += 1
            indptr.append(indptr[-1] + row_nnz)

        return cls(data, indices, indptr, (rows, cols))

    def _to_csc(self) -> 'CSCMatrix':
        return self.transpose()

    def _to_coo(self) -> 'COOMatrix':
        rows, cols = self.shape
        data = []
        row_indices = []
        col_indices = []

        for i in range(rows):
            start, end = self.indptr[i], self.indptr[i + 1]
            for pos in range(start, end):
                data.append(self.data[pos])
                row_indices.append(i)
                col_indices.append(self.indices[pos])

        return COOMatrix(data, row_indices, col_indices, self.shape)

    @classmethod
    def from_coo(cls, data: COOData, rows: COORows, cols: COOCols, shape: Shape) -> 'CSRMatrix':
        coo = COOMatrix(data, rows, cols, shape)
        return coo._to_csr()