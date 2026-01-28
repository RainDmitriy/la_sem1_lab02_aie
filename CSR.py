from base import Matrix
from matrix_types import CSRData, CSRIndices, CSRIndptr, Shape, DenseMatrix

class CSRMatrix(Matrix):
    def __init__(self, data: CSRData, indices: CSRIndices, indptr: CSRIndptr, shape: Shape):
        super().__init__(shape)
        self.data = data
        self.indices = indices
        self.indptr = indptr
        self.nnz = len(data)
        if len(indptr) != shape[0] + 1:
            raise ValueError(f"indptr должен иметь длину shape[0] + 1 = {shape[0] + 1}, получено {len(indptr)}")
        if indptr[-1] != len(data):
            raise ValueError(
                f"последний элемент indptr должен быть равен len(data) = {len(data)}, получено {indptr[-1]}")
        for i in range(len(indptr) - 1):
            if indptr[i] > indptr[i + 1]:
                raise ValueError(f"indptr не монотонен: indptr[{i}] = {indptr[i]} > indptr[{i + 1}] = {indptr[i + 1]}")
        for idx in indices:
            if idx < 0 or idx >= shape[1]:
                raise ValueError(f"индекс столбца {idx} вне диапазона [0, {shape[1] - 1}]")
        if len(indices) != len(data):
            raise ValueError(f"длины indices ({len(indices)}) и data ({len(data)}) не совпадают")

    def to_dense(self) -> DenseMatrix:
        """преобразует CSR в плотную матрицу"""
        rows, cols = self.shape
        dense = [[0.0] * cols for _ in range(rows)]
        for i in range(rows):
            start = self.indptr[i]
            end = self.indptr[i + 1]
            for idx in range(start, end):
                j = self.indices[idx]
                dense[i][j] = self.data[idx]

        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """сложение CSR матриц"""
        if self.shape != other.shape:
            raise ValueError("матрицы должны иметь одинаковые размеры")
        coo_self = self._to_coo()
        if hasattr(other, '_to_coo'):
            coo_other = other._to_coo()
        else:
            from COO import COOMatrix
            coo_other = COOMatrix.from_dense(other.to_dense())
        result_coo = coo_self._add_impl(coo_other)
        return result_coo._to_csr()


    def _mul_impl(self, scalar: float) -> 'Matrix':
        """умножение на скаляр"""
        new_data = [value * scalar for value in self.data]
        return CSRMatrix(new_data, self.indices.copy(), self.indptr.copy(), self.shape)

    def transpose(self) -> 'Matrix':
        """транспонирование матрицы"""
        return self._to_csc()

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        from CSR import CSRMatrix
        if self.shape[1] != other.shape[0]:
            raise ValueError("несовместимые размерности")
        if isinstance(other, CSRMatrix):
            from CSR import CSRMatrix
            result_rows = self.shape[0]
            result_cols = other.shape[1]
            result_data = []
            result_indices = []
            result_indptr = [0]

            for i in range(result_rows):
                row_dict = {}
                for idx1 in range(self.indptr[i], self.indptr[i + 1]):
                    k = self.indices[idx1]
                    val_self = self.data[idx1]
                    pass
            return self._matmul_general(other)
        else:
            return self._matmul_general(other)

    def _matmul_general(self, other: 'Matrix') -> 'CSRMatrix':
        """Общее умножение через оптимизированный алгоритм"""
        from CSC import CSCMatrix
        result_rows = self.shape[0]
        result_cols = other.shape[1]
        k_dim = self.shape[1]
        temp_result = [[0.0] * result_cols for _ in range(result_rows)]
        for i in range(result_rows):
            for idx in range(self.indptr[i], self.indptr[i + 1]):
                k = self.indices[idx]
                val_self = self.data[idx]
                if isinstance(other, CSRMatrix):
                    for idx2 in range(other.indptr[k], other.indptr[k + 1]):
                        j = other.indices[idx2]
                        temp_result[i][j] += val_self * other.data[idx2]
                elif isinstance(other, CSCMatrix):
                    other_dense = other.to_dense()
                    for j in range(result_cols):
                        temp_result[i][j] += val_self * other_dense[k][j]
                else:
                    other_dense = other.to_dense()
                    for j in range(result_cols):
                        temp_result[i][j] += val_self * other_dense[k][j]
        return CSRMatrix.from_dense(temp_result)

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSRMatrix':
        """создание CSR из плотной матрицы"""
        from COO import COOMatrix
        coo = COOMatrix.from_dense(dense_matrix)
        return coo._to_csr()

    def _to_csc(self) -> 'CSCMatrix':
        """преобразование CSRMatrix в CSCMatrix"""
        from CSC import CSCMatrix
        cols = self.shape[1]
        col_counts = [0] * cols
        for j in self.indices:
            col_counts[j] += 1
        csc_indptr = [0] * (cols + 1)
        for j in range(cols):
            csc_indptr[j + 1] = csc_indptr[j] + col_counts[j]
        csc_data = [0.0] * self.nnz
        csc_indices = [0] * self.nnz
        current_pos = csc_indptr.copy()
        for i in range(self.shape[0]):
            for idx in range(self.indptr[i], self.indptr[i + 1]):
                j = self.indices[idx]
                pos = current_pos[j]
                csc_data[pos] = self.data[idx]
                csc_indices[pos] = i
                current_pos[j] += 1

        return CSCMatrix(csc_data, csc_indices, csc_indptr, self.shape)

    def _to_coo(self) -> 'COOMatrix':
        """преобразование CSRMatrix в COOMatrix"""
        from COO import COOMatrix
        data = []
        rows = []
        cols = []
        for i in range(self.shape[0]):
            for idx in range(self.indptr[i], self.indptr[i + 1]):
                data.append(self.data[idx])
                rows.append(i)
                cols.append(self.indices[idx])

        return COOMatrix(data, rows, cols, self.shape)

    def __str__(self) -> str:
        return f"CSRMatrix(shape={self.shape}, nnz={self.nnz})"

    def __repr__(self) -> str:
        return self.__str__()