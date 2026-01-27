from base import Matrix
from mytypes import CSCData, CSCIndices, CSCIndptr, Shape, DenseMatrix

class CSCMatrix(Matrix):
    def __init__(self, data: CSCData, indices: CSCIndices, indptr: CSCIndptr, shape: Shape):
        super().__init__(shape)
        n, m = shape
        if len(indptr) != m + 1:
            raise ValueError(f"indptr должен иметь длину m+1 = {m+1}, получено {len(indptr)}")

        for j in range(m):
            if indptr[j] > indptr[j + 1]:
                raise ValueError(f"indptr должен быть неубывающим: indptr[{j}] = {indptr[j]} > indptr[{j+1}] = {indptr[j+1]}")

        if indptr[0] != 0:
            raise ValueError(f"indptr[0] должен быть 0, получено {indptr[0]}")
        if indptr[-1] != len(data):
            raise ValueError(f"indptr[-1] должен быть равен len(data) = {len(data)}, получено {indptr[-1]}")

        if len(data) != len(indices):
            raise ValueError(f"data и indices должны быть одинаковой длины: data={len(data)}, indices={len(indices)}")

        for row_idx in indices:
            if not (0 <= row_idx < n):
                raise ValueError(f"Индекс строки {row_idx} вне диапазона [0, {n-1}]")
        self.data = data
        self.indices = indices
        self.indptr = indptr
        self.nnz = len(data)

    def to_dense(self) -> DenseMatrix:
        """Преобразует CSC в плотную матрицу."""
        n, m = self.shape
        if n * m > 10000:
            raise MemoryError(f"Матрица {n}x{m} слишком большая для dense")
        
        mat = [[0.0] * m for _ in range(n)]
        for j in range(m):
            start = self.indptr[j]
            end = self.indptr[j + 1]
            for idx in range(start, end):
                i = self.indices[idx]
                mat[i][j] = self.data[idx]
        return mat

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение CSC матриц."""
        if hasattr(other, '_to_coo'):
            return self._to_coo()._add_impl(other)._to_csc()
        
        if self.shape[0] * self.shape[1] <= 10000:
            from COO import COOMatrix
            other_coo = COOMatrix.from_dense(other.to_dense())
            return self._add_impl(other_coo)
        
        raise ValueError("Нельзя складывать большие матрицы через dense")

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение CSC на скаляр."""
        new_data = [x * scalar for x in self.data]
        return CSCMatrix(new_data, self.indices, self.indptr, self.shape)

    def transpose(self) -> 'Matrix':
        """
        Транспонирование CSC матрицы.
        Hint:
        Результат - в CSR формате (с теми же данными, но с интерпретацией строк как столбцов).
        """
        from CSR import CSRMatrix
        
        n, m = self.shape
        nnz = self.nnz
        
        row_counts = [0] * n
        for row in self.indices:
            row_counts[row] += 1
        
        indptr_csr = [0] * (n + 1)
        for i in range(n):
            indptr_csr[i + 1] = indptr_csr[i] + row_counts[i]
        
        current_pos = indptr_csr.copy()
        data_csr = [0.0] * nnz
        indices_csr = [0] * nnz
        
        for j in range(m):
            for pos in range(self.indptr[j], self.indptr[j + 1]):
                i = self.indices[pos]
                csr_pos = current_pos[i]
                data_csr[csr_pos] = self.data[pos]
                indices_csr[csr_pos] = j
                current_pos[i] += 1
        
        return CSRMatrix(data_csr, indices_csr, indptr_csr, (m, n))

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение CSC матриц."""
        if isinstance(other, CSCMatrix):
            A_n, A_k = self.shape
            B_k, B_m = other.shape
            
            if A_k != B_k:
                raise ValueError("Несовместимые размерности для умножения")
            B_csr = other.transpose().transpose()
            
            result_data, result_indices, result_indptr = [], [], [0]
            
            temp_result = [{} for _ in range(A_n)]

            for j in range(B_m):
                for b_pos in range(B_csr.indptr[j], B_csr.indptr[j + 1]):
                    k = B_csr.indices[b_pos]
                    b_val = B_csr.data[b_pos]

                    for a_pos in range(self.indptr[k], self.indptr[k + 1]):
                        i = self.indices[a_pos]
                        a_val = self.data[a_pos]

                        temp_result[i][j] = temp_result[i].get(j, 0.0) + a_val * b_val

            nnz_count = 0
            for j in range(B_m):
                result_indptr.append(nnz_count)

                col_elements = []
                for i in range(A_n):
                    if j in temp_result[i]:
                        val = temp_result[i][j]
                        if abs(val) > 1e-12:
                            col_elements.append((i, val))

                col_elements.sort(key=lambda x: x[0])
                
                for i, val in col_elements:
                    result_indices.append(i)
                    result_data.append(val)
                    nnz_count += 1

            result_indptr.append(nnz_count)
            
            return CSCMatrix(result_data, result_indices, result_indptr, (A_n, B_m))

        if hasattr(other, '_to_coo'):
            return self._to_coo()._matmul_impl(other)._to_csc()

        if self.shape[0] * self.shape[1] <= 10000 and \
        other.shape[0] * other.shape[1] <= 10000:
            from COO import COOMatrix
            other_coo = COOMatrix.from_dense(other.to_dense())
            return self._matmul_impl(other_coo)
        
        raise ValueError("Нельзя умножать большие матрицы через dense")

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSCMatrix':
        """Создание CSC из плотной матрицы."""
        from COO import COOMatrix
        return COOMatrix.from_dense(dense_matrix)._to_csc()

    def _to_csr(self) -> 'CSRMatrix':
        """
        Преобразование CSCMatrix в CSRMatrix.
        """
        return self.transpose()

    def _to_coo(self) -> 'COOMatrix':
        """
        Преобразование CSCMatrix в COOMatrix.
        """
        from COO import COOMatrix
        
        data, rows, cols = [], [], []
        _, m = self.shape
        
        for j in range(m):
            start = self.indptr[j]
            end = self.indptr[j + 1]
            for idx in range(start, end):
                data.append(self.data[idx])
                rows.append(self.indices[idx])
                cols.append(j)
        
        return COOMatrix(data, rows, cols, self.shape)
