from base import Matrix
from type import CSRData, CSRIndices, CSRIndptr, Shape, DenseMatrix


class CSRMatrix(Matrix):
    def __init__(self, data: CSRData, indices: CSRIndices, indptr: CSRIndptr, shape: Shape):
        
        if not (isinstance(shape, tuple) and len(shape) == 2):
            raise TypeError("shape должен быть кортежем (n_rows, n_cols)")
        n_rows, n_cols = shape
        if n_rows < 0 or n_cols < 0:
            raise ValueError("Размерности shape должны быть неотрицательными")
        
        super().__init__(shape)

        self.data: CSRData = list(data)
        self.indices: CSRIndices = list(indices)
        self.indptr: CSRIndptr = list(indptr)

        nnz = len(self.data)
        if len(self.indices) != nnz:
            raise ValueError("data и indices должны быть одинаковой длины (nnz)")

        if len(self.indptr) != n_rows + 1:
            raise ValueError("indptr должен иметь длину n_rows + 1")

        if n_rows == 0:
            if self.indptr != [0]:
                raise ValueError("Для n_rows=0 indptr должен быть [0]")
            if nnz != 0:
                raise ValueError("Для n_rows=0 data/indices должны быть пустыми")
            return

        if self.indptr[0] != 0:
            raise ValueError("indptr[0] должен быть равен 0")
        if self.indptr[-1] != nnz:
            raise ValueError("indptr[-1] должен быть равен nnz")

        prev = 0
        for p in self.indptr:
            if not isinstance(p, int):
                raise TypeError("indptr должен содержать целые числа")
            if p < prev:
                raise ValueError("indptr должен быть неубывающим")
            if p < 0 or p > nnz:
                raise ValueError("Значения indptr должны быть в диапазоне [0, nnz]")
            prev = p

        # Проверка индексов столбцов
        for idx, j in enumerate(self.indices):
            if not isinstance(j, int):
                raise TypeError("indices должен содержать целые числа")
            if j < 0 or j >= n_cols:
                raise IndexError(f"indices[{idx}]={j} вне диапазона [0, {n_cols})")
            
            
    def to_dense(self) -> DenseMatrix:
        n_rows, n_cols = self.shape

        if n_rows == 0:
            return []
        if n_cols == 0:
            return [[] for _ in range(n_rows)]

        dense: DenseMatrix = [[0.0] * n_cols for _ in range(n_rows)]
        for i in range(n_rows):
            start = self.indptr[i]
            end = self.indptr[i + 1]
            for p in range(start, end):
                j = self.indices[p]
                dense[i][j] += float(self.data[p])
        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        n_rows, n_cols = self.shape

        if isinstance(other, CSRMatrix):
            data_out: CSRData = []
            indices_out: CSRIndices = []
            indptr_out: CSRIndptr = [0]

            for i in range(n_rows):
                acc: dict[int, float] = {}

                a0, a1 = self.indptr[i], self.indptr[i + 1]
                for p in range(a0, a1):
                    j = self.indices[p]
                    v = float(self.data[p])
                    if v != 0.0:
                        acc[j] = acc.get(j, 0.0) + v

                b0, b1 = other.indptr[i], other.indptr[i + 1]
                for p in range(b0, b1):
                    j = other.indices[p]
                    v = float(other.data[p])
                    if v != 0.0:
                        acc[j] = acc.get(j, 0.0) + v

                items = [(j, v) for j, v in acc.items() if v != 0.0]
                items.sort(key=lambda t: t[0])

                for j, v in items:
                    indices_out.append(j)
                    data_out.append(v)

                indptr_out.append(len(data_out))

            return CSRMatrix(data_out, indices_out, indptr_out, (n_rows, n_cols))

        b = other.to_dense()
        dense = self.to_dense()
        for i in range(n_rows):
            di = dense[i]
            bi = b[i]
            for j in range(n_cols):
                di[j] += float(bi[j])
        return CSRMatrix.from_dense(dense)
    

    def _mul_impl(self, scalar: float) -> 'Matrix':
        scalar = float(scalar)
        n_rows, n_cols = self.shape

        data_out: CSRData = [float(v) * scalar for v in self.data]
        return CSRMatrix(data_out, list(self.indices), list(self.indptr), (n_rows, n_cols))
    

    def transpose(self) -> 'Matrix':
        csc = self._to_csc()
        csc.shape = (self.shape[1], self.shape[0])
        return csc
        

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        n_rows, k = self.shape
        _, n_cols_out = other.shape

        if n_rows == 0:
            return CSRMatrix([], [], [0], (0, n_cols_out))
        if n_cols_out == 0:
            return CSRMatrix([], [], [0] * (n_rows + 1), (n_rows, 0))
        if k == 0 or len(self.data) == 0:
            return CSRMatrix([], [], [0] * (n_rows + 1), (n_rows, n_cols_out))

        data_out: list[float] = []
        indices_out: list[int] = []
        indptr_out: list[int] = [0]

        if isinstance(other, CSRMatrix):
            for i in range(n_rows):
                acc: dict[int, float] = {}

                a0, a1 = self.indptr[i], self.indptr[i + 1]
                for p in range(a0, a1):
                    t = self.indices[p]
                    a_it = float(self.data[p])
                    if a_it == 0.0:
                        continue

                    b0, b1 = other.indptr[t], other.indptr[t + 1]
                    for q in range(b0, b1):
                        j = other.indices[q]
                        acc[j] = acc.get(j, 0.0) + a_it * float(other.data[q])

                items = [(j, v) for j, v in acc.items() if v != 0.0]
                items.sort(key=lambda t: t[0])

                for j, v in items:
                    indices_out.append(j)
                    data_out.append(v)

                indptr_out.append(len(data_out))

            return CSRMatrix(data_out, indices_out, indptr_out, (n_rows, n_cols_out))
        
        b = other.to_dense()
        for i in range(n_rows):
            acc: dict[int, float] = {}
            a0, a1 = self.indptr[i], self.indptr[i + 1]

            for p in range(a0, a1):
                t = self.indices[p]
                a_it = float(self.data[p])
                if a_it == 0.0:
                    continue

                bt = b[t]
                for j in range(n_cols_out):
                    b_tj = float(bt[j])
                    if b_tj != 0.0:
                        acc[j] = acc.get(j, 0.0) + a_it * b_tj

                items = [(j, v) for j, v in acc.items() if v != 0.0]
                items.sort(key=lambda t: t[0])

                for j, v in items:
                    indices_out.append(j)
                    data_out.append(v)

                indptr_out.append(len(data_out))

        return CSRMatrix(data_out, indices_out, indptr_out, (n_rows, n_cols_out))

    

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSRMatrix':
        n_rows = len(dense_matrix)
        if n_rows == 0:
            return cls([], [], [0], (0, 0))

        n_cols = len(dense_matrix[0])
        if any(len(row) != n_cols for row in dense_matrix):
            raise ValueError("Плотная матрица должна быть прямоугольной")

        if n_cols == 0:
            return cls([], [], [0] * (n_rows + 1), (n_rows, 0))

        data: CSRData = []
        indices: CSRIndices = []
        indptr: CSRIndptr = [0]

        for i in range(n_rows):
            for j in range(n_cols):
                v = float(dense_matrix[i][j])
                if v != 0.0:
                    data.append(v)
                    indices.append(j)
            indptr.append(len(data))

        return cls(data, indices, indptr, (n_rows, n_cols))
    

    def _to_csc(self) -> 'CSCMatrix':
        from CSC import CSCMatrix

        n_rows, n_cols = self.shape
        nnz = len(self.data)

        if n_cols == 0:
            return CSCMatrix([], [], [0], (n_rows, 0))
        if n_rows == 0:
            return CSCMatrix([], [], [0] * (n_cols + 1), (0, n_cols))
        if nnz == 0:
            return CSCMatrix([], [], [0] * (n_cols + 1), (n_rows, n_cols))

        col_counts = [0] * n_cols
        for j in self.indices:
            col_counts[j] += 1

        indptr_out: list[int] = [0] * (n_cols + 1)
        for j in range(n_cols):
            indptr_out[j + 1] = indptr_out[j] + col_counts[j]

        next_pos = indptr_out[:]  
        data_out: list[float] = [0.0] * nnz
        indices_out: list[int] = [0] * nnz  

        for i in range(n_rows):
            start, end = self.indptr[i], self.indptr[i + 1]
            for p in range(start, end):
                j = self.indices[p]
                pos = next_pos[j]
                data_out[pos] = float(self.data[p])
                indices_out[pos] = i
                next_pos[j] += 1

        return CSCMatrix(data_out, indices_out, indptr_out, (n_rows, n_cols))
    
    
    def _to_coo(self) -> 'COOMatrix':
        from COO import COOMatrix 

        n_rows, n_cols = self.shape

        data_out = [float(v) for v in self.data]
        col_out: list[int] = list(self.indices)
        row_out: list[int] = []

        for i in range(n_rows):
            start, end = self.indptr[i], self.indptr[i + 1]
            row_out.extend([i] * (end - start))

        return COOMatrix(data_out, row_out, col_out, (n_rows, n_cols))