from base import Matrix
from type import CSRData, CSRIndices, CSRIndptr, Shape, DenseMatrix


class CSRMatrix(Matrix):
    def __init__(self, data: CSRData, indices: CSRIndices, indptr: CSRIndptr, shape: Shape):
        super().__init__(shape)
        self.data = data.copy()
        self.indices = indices.copy()
        self.indptr = indptr.copy()

    def to_dense(self) -> DenseMatrix:
        rows, cols = self.shape
        dense = [[0.0] * cols for _ in range(rows)]
        for i in range(rows):
            for idx in range(self.indptr[i], self.indptr[i + 1]):
                j = self.indices[idx]
                dense[i][j] = self.data[idx]
        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        if isinstance(other, CSRMatrix):
            rows, cols = self.shape
            result_data = []
            result_indices = []
            result_indptr = [0]

            for i in range(rows):
                idx1 = self.indptr[i]
                idx2 = other.indptr[i]
                end1 = self.indptr[i + 1]
                end2 = other.indptr[i + 1]

                while idx1 < end1 and idx2 < end2:
                    col1 = self.indices[idx1]
                    col2 = other.indices[idx2]

                    if col1 < col2:
                        result_data.append(self.data[idx1])
                        result_indices.append(col1)
                        idx1 += 1
                    elif col1 > col2:
                        result_data.append(other.data[idx2])
                        result_indices.append(col2)
                        idx2 += 1
                    else:
                        val = self.data[idx1] + other.data[idx2]
                        if abs(val) > 1e-12:
                            result_data.append(val)
                            result_indices.append(col1)
                        idx1 += 1
                        idx2 += 1

                while idx1 < end1:
                    result_data.append(self.data[idx1])
                    result_indices.append(self.indices[idx1])
                    idx1 += 1

                while idx2 < end2:
                    result_data.append(other.data[idx2])
                    result_indices.append(other.indices[idx2])
                    idx2 += 1

                result_indptr.append(len(result_data))

            return CSRMatrix(result_data, result_indices, result_indptr, self.shape)
        else:
            from COO import COOMatrix
            coo_self = self._to_coo()
            coo_other = COOMatrix.from_dense(other.to_dense())
            return coo_self._add_impl(coo_other)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        if scalar == 0:
            return CSRMatrix([], [], [0] * (self.shape[0] + 1), self.shape)
        new_data = [val * scalar for val in self.data]
        return CSRMatrix(new_data, self.indices.copy(), self.indptr.copy(), self.shape)

    def transpose(self) -> 'Matrix':
        from CSC import CSCMatrix

        if len(self.data) == 0:
            return CSCMatrix([], [], [0] * (self.shape[1] + 1), (self.shape[1], self.shape[0]))

        rows, cols = self.shape
        nnz = len(self.data)

        data_csc = [0.0] * nnz
        indices_csc = [0] * nnz
        indptr_csc = [0] * (cols + 1)

        for j in self.indices:
            indptr_csc[j + 1] += 1

        for j in range(1, cols + 1):
            indptr_csc[j] += indptr_csc[j - 1]

        current_pos = indptr_csc.copy()
        for i in range(rows):
            for idx in range(self.indptr[i], self.indptr[i + 1]):
                j = self.indices[idx]
                pos = current_pos[j]
                data_csc[pos] = self.data[idx]
                indices_csc[pos] = i
                current_pos[j] += 1

        return CSCMatrix(data_csc, indices_csc, indptr_csc, (cols, rows))

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        if isinstance(other, CSRMatrix):
            A_rows, A_cols = self.shape
            B_rows, B_cols = other.shape

            if A_cols != B_rows:
                raise ValueError("Несовместимые размерности для умножения")

            B_T = other.transpose()

            result_data = []
            result_indices = []
            result_indptr = [0]

            for i in range(A_rows):
                row_dict = {}

                for a_idx in range(self.indptr[i], self.indptr[i + 1]):
                    k = self.indices[a_idx]
                    a_val = self.data[a_idx]

                    if k < B_T.shape[0]:
                        for b_idx in range(B_T.indptr[k], B_T.indptr[k + 1]):
                            j = B_T.indices[b_idx]
                            b_val = B_T.data[b_idx]
                            row_dict[j] = row_dict.get(j, 0.0) + a_val * b_val

                sorted_cols = sorted(row_dict.keys())
                for j in sorted_cols:
                    val = row_dict[j]
                    if abs(val) > 1e-12:
                        result_data.append(val)
                        result_indices.append(j)

                result_indptr.append(len(result_data))

            return CSRMatrix(result_data, result_indices, result_indptr, (A_rows, B_cols))
        else:
            from COO import COOMatrix
            coo_self = self._to_coo()
            coo_other = COOMatrix.from_dense(other.to_dense())
            return coo_self._matmul_impl(coo_other)

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSRMatrix':
        rows = len(dense_matrix)
        cols = len(dense_matrix[0]) if rows > 0 else 0
        data = []
        indices = []
        indptr = [0]

        for i in range(rows):
            for j in range(cols):
                val = dense_matrix[i][j]
                if val != 0:
                    data.append(val)
                    indices.append(j)
            indptr.append(len(data))

        return cls(data, indices, indptr, (rows, cols))

    def _to_csc(self) -> 'CSCMatrix':
        return self.transpose()

    def _to_coo(self) -> 'COOMatrix':
        from COO import COOMatrix

        rows, cols = self.shape
        data = []
        row_indices = []
        col_indices = []

        for i in range(rows):
            for idx in range(self.indptr[i], self.indptr[i + 1]):
                data.append(self.data[idx])
                row_indices.append(i)
                col_indices.append(self.indices[idx])

        return COOMatrix(data, row_indices, col_indices, self.shape)