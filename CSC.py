from base import Matrix
from type import CSCData, CSCIndices, CSCIndptr, Shape, DenseMatrix

class CSCMatrix(Matrix):
    def __init__(self, data: CSCData, indices: CSCIndices, indptr: CSCIndptr, shape: Shape):
        super().__init__(shape)
        if indptr[0] != 0:
            raise ValueError("Первый элемент indptr должен быть равен 0")

        self.data = data


    def _to_csc(self) -> 'CSCMatrix':
        """CSC -> CSC (просто возвращаем себя)"""
        return self
    def __init__(self, data: CSCData, indices: CSCIndices, indptr: CSCIndptr, shape: Shape):
        super().__init__(shape)
        self.data = data
        self.indices = indices
        self.indptr = indptr
        self.nnz = len(data)
        if len(indptr) != shape[1] + 1:
            raise ValueError(f"indptr должен иметь длину shape[1] + 1 = {shape[1] + 1}, получено {len(indptr)}")
        if indptr[-1] != len(data):
            raise ValueError(
                f"последний элемент indptr должен быть равен len(data) = {len(data)}, получено {indptr[-1]}")
        for i in range(len(indptr) - 1):
            if indptr[i] > indptr[i + 1]:
                raise ValueError(f"indptr не монотонен: indptr[{i}] = {indptr[i]} > indptr[{i + 1}] = {indptr[i + 1]}")
        for idx in indices:
            if idx < 0 or idx >= shape[0]:
                raise ValueError(f"индекс строки {idx} вне диапазона [0, {shape[0] - 1}]")
        if len(indices) != len(data):
            raise ValueError(f"Длины indices ({len(indices)}) и data ({len(data)}) не совпадают")

    def to_dense(self) -> DenseMatrix:
        """преобразует CSC в плотную матрицу"""
        rows, cols = self.shape
        dense = [[0.0] * cols for _ in range(rows)]
        for j in range(cols):
            start = self.indptr[j]
            end = self.indptr[j + 1]
            for idx in range(start, end):
                i = self.indices[idx]
                dense[i][j] = self.data[idx]

        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """сложение CSC матриц"""
        if self.shape != other.shape:
            raise ValueError("матрицы должны иметь одинаковые размеры")
        coo_self = self._to_coo()
        if hasattr(other, '_to_coo'):
            coo_other = other._to_coo()
        else:
            from COO import COOMatrix
            coo_other = COOMatrix.from_dense(other.to_dense())
        result_coo = coo_self._add_impl(coo_other)
        return result_coo._to_csc()

    def _mul_impl(self, scalar: float) -> 'Matrix':
        if abs(scalar) < 1e-14:
            return CSCMatrix([], [], [0] * (self.shape[1] + 1), self.shape)
        data = [float(d) * float(scalar) for d in self.data]
        return CSCMatrix(data, self.indices[:], self.indptr[:], self.shape)

    def transpose(self) -> 'Matrix':
        """транспонирование матрицы"""
        return self._to_csr()

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение CSC матриц"""
        a = self._to_csr()
        c = a._matmul_impl(other)

        if hasattr(c, "_to_csc"):
            return c._to_csc()
        from COO import COOMatrix
        return COOMatrix.from_dense(c.to_dense())._to_csc()


    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSCMatrix':
        """Создание CSC из плотной матрицы"""
        from COO import COOMatrix
        coo = COOMatrix.from_dense(dense_matrix)
        return coo._to_csc()

    def _to_csr(self) -> 'CSRMatrix':
        """преобразование CSC в CSR"""
        from CSR import CSRMatrix
        rows = self.shape[0]
        row_counts = [0] * rows
        for i in self.indices:
            row_counts[i] += 1
        csr_indptr = [0] * (rows + 1)
        for i in range(rows):
            csr_indptr[i + 1] = csr_indptr[i] + row_counts[i]
        csr_data = [0.0] * self.nnz
        csr_indices = [0] * self.nnz
        current_pos = csr_indptr.copy()
        for j in range(self.shape[1]):
            for idx in range(self.indptr[j], self.indptr[j + 1]):
                i = self.indices[idx]
                pos = current_pos[i]
                csr_data[pos] = self.data[idx]
                csr_indices[pos] = j
                current_pos[i] += 1

        return CSRMatrix(csr_data, csr_indices, csr_indptr, self.shape)

    def _to_coo(self) -> 'COOMatrix':
        """преобразование CSC в COO"""
        from COO import COOMatrix
        data = []
        rows = []
        cols = []

        for j in range(self.shape[1]):
            for idx in range(self.indptr[j], self.indptr[j + 1]):
                data.append(float(self.data[idx]))
                rows.append(int(self.indices[idx]))
                cols.append(j)
        return COOMatrix(data, rows, cols, self.shape)

    def __str__(self) -> str:
        return f"CSCMatrix(shape={self.shape}, nnz={self.nnz})"

    def __repr__(self) -> str:
        return self.__str__()