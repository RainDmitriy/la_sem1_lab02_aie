from base import Matrix
from types import CSCData, CSCIndices, CSCIndptr, Shape, DenseMatrix


class CSCMatrix(Matrix):
    def __init__(self, data: CSCData, indices: CSCIndices, indptr: CSCIndptr, shape: Shape):
        super().__init__(shape)
        self.data = list(data)
        self.indices = list(indices)
        self.indptr = list(indptr)

    def to_dense(self) -> DenseMatrix:
        """Преобразует CSC в плотную матрицу."""
        n, m = self.shape
        res = [[0.0] * m for _ in range(n)]
        
        # В CSC indptr идет по столбцам
        for j in range(m):
            start_index = self.indptr[j]
            end_index = self.indptr[j+1]
            
            for k in range(start_index, end_index):
                row_idx = self.indices[k]
                val = self.data[k]
                res[row_idx][j] = val
        return res

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение CSC матриц."""
        my_coo = self._to_coo()
        
        if hasattr(other, '_to_coo'):
            other_coo = other._to_coo()
        else:
            from COO import COOMatrix
            other_coo = COOMatrix.from_dense(other.to_dense())
            
        res_coo = my_coo._add_impl(other_coo)
        return res_coo._to_csc()    

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение CSC на скаляр."""
        new_data = []
        for x in self.data:
            new_data.append(x * scalar)
        return CSCMatrix(new_data, list(self.indices), list(self.indptr), self.shape)

    def transpose(self) -> 'Matrix':
        """
        Транспонирование CSC матрицы.
        Hint:
        Результат - в CSR формате (с теми же данными, но с интерпретацией строк как столбцов).
        """
        from CSR import CSRMatrix
        return CSRMatrix(
            list(self.data), 
            list(self.indices), 
            list(self.indptr), 
            (self.shape[1], self.shape[0])
        )

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение CSC матриц."""
        return self._to_coo()._matmul_impl(other)._to_csc()

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSCMatrix':
        """Создание CSC из плотной матрицы."""
        from COO import COOMatrix
        return COOMatrix.from_dense(dense_matrix)._to_csc()

    def _to_csr(self) -> 'CSRMatrix':
        """
        Преобразование CSCMatrix в CSRMatrix.
        """
        return self._to_coo()._to_csr()

    def _to_coo(self) -> 'COOMatrix':
        """
        Преобразование CSCMatrix в COOMatrix.
        """
        from COO import COOMatrix
        coo_cols = []
        for j in range(self.shape[1]):
            count = self.indptr[j+1] - self.indptr[j]
            for _ in range(count):
                coo_cols.append(j)
                
        return COOMatrix(self.data, self.indices, coo_cols, self.shape)
