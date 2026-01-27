from base import Matrix
from mytypes import CSRData, CSRIndices, CSRIndptr, Shape, DenseMatrix
from collections import defaultdict
from typing import List, Tuple

class CSRMatrix(Matrix):
    def __init__(self, data: CSRData, indices: CSRIndices, indptr: CSRIndptr, shape: Shape):
        super().__init__(shape)
        n, m = shape
        if len(indptr) != n + 1:
            raise ValueError(f"indptr должен иметь длину n+1 = {n+1}, получено {len(indptr)}")

        for i in range(n):
            if indptr[i] > indptr[i + 1]:
                raise ValueError(f"indptr должен быть неубывающим: indptr[{i}] = {indptr[i]} > indptr[{i+1}] = {indptr[i+1]}")

        if indptr[0] != 0:
            raise ValueError(f"indptr[0] должен быть 0, получено {indptr[0]}")
        if indptr[-1] != len(data):
            raise ValueError(f"indptr[-1] должен быть равен len(data) = {len(data)}, получено {indptr[-1]}")

        if len(data) != len(indices):
            raise ValueError(f"data и indices должны быть одинаковой длины: data={len(data)}, indices={len(indices)}")

        for col_idx in indices:
            if not (0 <= col_idx < m):
                raise ValueError(f"Индекс столбца {col_idx} вне диапазона [0, {m-1}]")
        self.data = data
        self.indices = indices
        self.indptr = indptr
        self.nnz = len(data)

    def to_dense(self) -> DenseMatrix:
        """Преобразует CSR в плотную матрицу."""
        n, m = self.shape
        if n * m > 10000:
            raise MemoryError(f"Матрица {n}x{m} слишком большая для dense")
        
        mat = [[0.0] * m for _ in range(n)]
        for i in range(n):
            start = self.indptr[i]
            end = self.indptr[i + 1]
            for idx in range(start, end):
                j = self.indices[idx]
                mat[i][j] = self.data[idx]
        return mat

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение CSR матриц."""
        ZERO_THRESHOLD = 1e-12

        if not isinstance(other, CSRMatrix):
            if hasattr(other, '_to_csr'):
                other = other._to_csr()
            else:
                n, m = self.shape
                if n * m <= 10000:
                    from COO import COOMatrix
                    other_coo = COOMatrix.from_dense(other.to_dense())
                    other = other_coo._to_csr()
                else:
                    raise ValueError("Нельзя складывать большие матрицы через dense")

        if self.shape != other.shape:
            raise ValueError(f"Размеры не совпадают: {self.shape} != {other.shape}")
        
        n, m = self.shape

        result_data = []
        result_indices = []
        result_indptr = [0]
        
        for i in range(n):
            p1 = self.indptr[i]
            p2 = other.indptr[i]
            
            end1 = self.indptr[i + 1]
            end2 = other.indptr[i + 1]

            while p1 < end1 and p2 < end2:
                j1 = self.indices[p1]
                j2 = other.indices[p2]
                
                if j1 < j2:
                    result_data.append(self.data[p1])
                    result_indices.append(j1)
                    p1 += 1
                    
                elif j1 > j2:
                    result_data.append(other.data[p2])
                    result_indices.append(j2)
                    p2 += 1
                    
                else:
                    sum_val = self.data[p1] + other.data[p2]
                    
                    if abs(sum_val) > ZERO_THRESHOLD:
                        result_data.append(sum_val)
                        result_indices.append(j1)
                    
                    p1 += 1
                    p2 += 1

            while p1 < end1:
                result_data.append(self.data[p1])
                result_indices.append(self.indices[p1])
                p1 += 1

            while p2 < end2:
                result_data.append(other.data[p2])
                result_indices.append(other.indices[p2])
                p2 += 1

            result_indptr.append(len(result_data))

        if len(result_indptr) != n + 1:
            raise ValueError(f"Неправильная длина indptr: {len(result_indptr)} != {n + 1}")
        
        if result_indptr[-1] != len(result_data):
            raise ValueError(
                f"Несоответствие: indptr[-1]={result_indptr[-1]}, "
                f"len(data)={len(result_data)}"
            )
        
        return CSRMatrix(result_data, result_indices, result_indptr, self.shape)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение CSR на скаляр."""
        new_data = [x * scalar for x in self.data]
        return CSRMatrix(new_data, self.indices, self.indptr, self.shape)

    def transpose(self) -> 'Matrix':
        """
        Транспонирование CSR матрицы.
        Возвращает CSC матрицу той же формы (m, n).
        """
        from CSC import CSCMatrix
        
        n, m = self.shape
        nnz = self.nnz

        if nnz == 0:
            return CSCMatrix([], [], [0] * (m + 1), (m, n))

        col_counts = [0] * m
        
        for i in range(n):
            for pos in range(self.indptr[i], self.indptr[i + 1]):
                j = self.indices[pos]
                col_counts[j] += 1

        csc_indptr = [0] * (m + 1)
        for j in range(m):
            csc_indptr[j + 1] = csc_indptr[j] + col_counts[j]

        csc_data = [0.0] * nnz
        csc_indices = [0] * nnz

        current_pos = csc_indptr.copy()

        for i in range(n):
            for pos in range(self.indptr[i], self.indptr[i + 1]):
                j = self.indices[pos]
                value = self.data[pos]

                csc_pos = current_pos[j]

                csc_data[csc_pos] = value
                csc_indices[csc_pos] = i

                current_pos[j] += 1

        if current_pos != csc_indptr[1:]:
            for j in range(m):
                if current_pos[j] != csc_indptr[j + 1]:
                    print(f"Warning: Столбец {j}: размещено {current_pos[j] - csc_indptr[j]}, "
                        f"ожидалось {col_counts[j]}")

        return CSCMatrix(csc_data, csc_indices, csc_indptr, (m, n))

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """Умножение CSR матриц."""
        ZERO_THRESHOLD = 1e-12

        if not isinstance(other, CSRMatrix):
            if hasattr(other, '_to_csr'):
                other_csr = other._to_csr()
            else:
                if self.shape[0] * self.shape[1] <= 10000 and \
                other.shape[0] * other.shape[1] <= 10000:
                    from COO import COOMatrix
                    other_coo = COOMatrix.from_dense(other.to_dense())
                    other_csr = other_coo._to_csr()
                else:
                    raise ValueError("Нельзя умножать большие матрицы через dense")
        else:
            other_csr = other
        
        n, k1 = self.shape
        k2, m = other_csr.shape
        
        if k1 != k2:
            raise ValueError(f"Несовместимые размерности: {self.shape} @ {other_csr.shape}")

        other_csc = other_csr.transpose()

        result_data = []
        result_indices = []
        result_indptr = [0]
        
        for i in range(n):
            row_accumulator = {}

            for a_pos in range(self.indptr[i], self.indptr[i + 1]):
                k = self.indices[a_pos]
                a_val = self.data[a_pos]

                for b_pos in range(other_csc.indptr[k], other_csc.indptr[k + 1]):
                    j = other_csc.indices[b_pos]
                    b_val = other_csc.data[b_pos]

                    product = a_val * b_val
                    if j in row_accumulator:
                        row_accumulator[j] += product
                    else:
                        row_accumulator[j] = product

            filtered_items = []
            for j, value in row_accumulator.items():
                if abs(value) > ZERO_THRESHOLD:
                    filtered_items.append((j, value))

            filtered_items.sort(key=lambda x: x[0])

            for j, value in filtered_items:
                result_indices.append(j)
                result_data.append(value)

            result_indptr.append(len(result_data))

        if len(result_indptr) != n + 1:
            raise ValueError(f"Неправильная длина indptr: {len(result_indptr)} != {n + 1}")
        
        return CSRMatrix(result_data, result_indices, result_indptr, (n, m))

    """Методы для LU"""
    def get_row(self, i: int) -> List[Tuple[int, float]]:
        start = self.indptr[i]
        end = self.indptr[i + 1]
        return [(self.indices[idx], self.data[idx]) for idx in range(start, end)]

    def set_row(self, i: int, elements: List[Tuple[int, float]]):
        n, m = self.shape
        
        all_rows = []
        for row_idx in range(n):
            if row_idx == i:
                all_rows.append(sorted(elements, key=lambda x: x[0]))
            else:
                all_rows.append(self.get_row(row_idx))
        
        new_data, new_indices, new_indptr = [], [], [0]
        
        for row in all_rows:
            for col, val in row:
                if abs(val) > 1e-12:
                    new_data.append(val)
                    new_indices.append(col)
            new_indptr.append(len(new_data))
        
        self.data = new_data
        self.indices = new_indices
        self.indptr = new_indptr
        self.nnz = len(new_data)

    def swap_rows(self, i: int, j: int):
        if i == j:
            return
        
        row_i = self.get_row(i)
        row_j = self.get_row(j)
        
        self.set_row(i, row_j)
        self.set_row(j, row_i)

    def get_element(self, i: int, j: int) -> float:
        start = self.indptr[i]
        end = self.indptr[i + 1]
        
        left, right = start, end - 1
        while left <= right:
            mid = (left + right) // 2
            col = self.indices[mid]
            if col == j:
                return self.data[mid]
            elif col < j:
                left = mid + 1
            else:
                right = mid - 1
        return 0.0

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSRMatrix':
        """Создание CSR из плотной матрицы."""
        from COO import COOMatrix
        return COOMatrix.from_dense(dense_matrix)._to_csr()

    def _to_csc(self) -> 'CSCMatrix':
        """
        Преобразование CSRMatrix в CSCMatrix.
        """
        return self.transpose()

    def _to_coo(self) -> 'COOMatrix':
        """
        Преобразование CSRMatrix в COOMatrix.
        """
        from COO import COOMatrix
        
        n, m = self.shape
        data, row_indices, col_indices = [], [], []
        
        for i in range(n):
            start = self.indptr[i]
            end = self.indptr[i + 1]
            for idx in range(start, end):
                data.append(self.data[idx])
                row_indices.append(i)
                col_indices.append(self.indices[idx])
        
        return COOMatrix(data, row_indices, col_indices, self.shape)
    