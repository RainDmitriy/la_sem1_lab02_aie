from base import Matrix
from type import CSCData, CSCIndices, CSCIndptr, Shape, DenseMatrix


class CSCMatrix(Matrix):
    def __init__(self, data: CSCData, indices: CSCIndices, indptr: CSCIndptr, shape: Shape):
        
        if not (isinstance(shape, tuple) and len(shape) == 2):
            raise TypeError("shape должен быть кортежем (n_rows, n_cols)")
        n_rows, n_cols = shape
        if n_rows < 0 or n_cols < 0:
            raise ValueError("Размерности shape должны быть неотрицательными")
        
        super().__init__(shape)

        self.data: CSCData = list(data)
        self.indices: CSCIndices = list(indices)
        self.indptr: CSCIndptr = list(indptr)

        nnz = len(self.data)
        if len(self.indices) != nnz:
            raise ValueError("data и indices должны быть одинаковой длины")

        if len(self.indptr) != n_cols + 1:
            raise ValueError("indptr должен иметь длину n_cols + 1")

        if n_cols == 0:
            if self.indptr != [0]:
                raise ValueError("Для n_cols=0 indptr должен быть [0]")
            if nnz != 0:
                raise ValueError("Для n_cols=0 data/indices должны быть пустыми")
            return

        if self.indptr[0] != 0:
            raise ValueError("indptr[0] должен быть равен 0")
        if self.indptr[-1] != nnz:
            raise ValueError("indptr[-1] должен быть равен nnz")

        prev = 0
        for j, p in enumerate(self.indptr):
            if not isinstance(p, int):
                raise TypeError("indptr должен содержать целые числа")
            if p < prev:
                raise ValueError("indptr должен быть неубывающим")
            if p < 0 or p > nnz:
                raise ValueError("Значения indptr должны быть в диапазоне [0, nnz]")
            prev = p

        for idx, i in enumerate(self.indices):
            if not isinstance(i, int):
                raise TypeError("indices должен содержать целые числа")
            if i < 0 or i >= n_rows:
                raise IndexError(f"indices[{idx}]={i} вне диапазона [0, {n_rows})")
            

    def to_dense(self) -> DenseMatrix:
        n_rows, n_cols = self.shape

        if n_rows == 0:
            return []
        if n_cols == 0:
            return [[] for _ in range(n_rows)]

        dense: DenseMatrix = [[0.0] * n_cols for _ in range(n_rows)]
        for j in range(n_cols):
            start = self.indptr[j]
            end = self.indptr[j + 1]
            for p in range(start, end):
                i = self.indices[p]
                dense[i][j] += float(self.data[p])
        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        n_rows, n_cols = self.shape

        if isinstance(other, CSCMatrix):
            data_out: CSCData = []
            indices_out: CSCIndices = []
            indptr_out: CSCIndptr = [0]

            for j in range(n_cols):
                acc: dict[int, float] = {}

                # self column j
                a0, a1 = self.indptr[j], self.indptr[j + 1]
                for p in range(a0, a1):
                    i = self.indices[p]
                    v = float(self.data[p])
                    if v != 0.0:
                        acc[i] = acc.get(i, 0.0) + v

                # other column j
                b0, b1 = other.indptr[j], other.indptr[j + 1]
                for p in range(b0, b1):
                    i = other.indices[p]
                    v = float(other.data[p])
                    if v != 0.0:
                        acc[i] = acc.get(i, 0.0) + v

                items = [(i, v) for i, v in acc.items() if v != 0.0]
                items.sort(key=lambda t: t[0])

                for i, v in items:
                    indices_out.append(i)
                    data_out.append(v)

                indptr_out.append(len(data_out))

            return CSCMatrix(data_out, indices_out, indptr_out, (n_rows, n_cols))

        b = other.to_dense()
        dense = self.to_dense()
        for i in range(n_rows):
            di = dense[i]
            bi = b[i]
            for j in range(n_cols):
                di[j] += float(bi[j])
        return CSCMatrix.from_dense(dense)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        scalar = float(scalar)
        n_rows, n_cols = self.shape

        if scalar == 0.0 or len(self.data) == 0:
            return CSCMatrix([], [], [0] * (n_cols + 1), (n_rows, n_cols))

        data_out: CSCData = []
        indices_out: CSCIndices = []
        indptr_out: CSCIndptr = [0]

        for j in range(n_cols):
            start, end = self.indptr[j], self.indptr[j + 1]
            for p in range(start, end):
                v = float(self.data[p]) * scalar
                if v != 0.0:
                    data_out.append(v)
                    indices_out.append(self.indices[p])
            indptr_out.append(len(data_out))

        return CSCMatrix(data_out, indices_out, indptr_out, (n_rows, n_cols))

    def transpose(self) -> 'Matrix':
        """
        Транспонирование CSC матрицы.
        Hint:
        Результат - в CSR формате (с теми же данными, но с интерпретацией строк как столбцов).
        """
        from CSR import CSRMatrix

        n_rows, n_cols = self.shape
        return CSRMatrix(
            list(self.data),
            list(self.indices),
            list(self.indptr),
            (n_cols, n_rows),
        )

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        n_rows, k = self.shape
        _, n_cols_out = other.shape

        if n_rows == 0:
            return CSCMatrix([], [], [0] * (n_cols_out + 1), (0, n_cols_out))
        if n_cols_out == 0:
            return CSCMatrix([], [], [0] * (0 + 1), (n_rows, 0))
        if k == 0 or len(self.data) == 0:
            return CSCMatrix([], [], [0] * (n_cols_out + 1), (n_rows, n_cols_out))

        data_out: CSCData = []
        indices_out: CSCIndices = []
        indptr_out: CSCIndptr = [0]

        if isinstance(other, CSCMatrix):
            for j in range(n_cols_out):
                acc: dict[int, float] = {}

                b0, b1 = other.indptr[j], other.indptr[j + 1]
                for p in range(b0, b1):
                    t = other.indices[p]       
                    b_tj = float(other.data[p])
                    if b_tj == 0.0:
                        continue

                    a0, a1 = self.indptr[t], self.indptr[t + 1]
                    for q in range(a0, a1):
                        i = self.indices[q]
                        acc[i] = acc.get(i, 0.0) + float(self.data[q]) * b_tj

                items = [(i, v) for i, v in acc.items() if v != 0.0]
                items.sort(key=lambda t: t[0])

                for i, v in items:
                    indices_out.append(i)
                    data_out.append(v)

                indptr_out.append(len(data_out))

            return CSCMatrix(data_out, indices_out, indptr_out, (n_rows, n_cols_out))

        b = other.to_dense()
        for j in range(n_cols_out):
            acc: dict[int, float] = {}
            for t in range(k):
                b_tj = float(b[t][j])
                if b_tj == 0.0:
                    continue

                a0, a1 = self.indptr[t], self.indptr[t + 1]
                for q in range(a0, a1):
                    i = self.indices[q]
                    acc[i] = acc.get(i, 0.0) + float(self.data[q]) * b_tj

            items = [(i, v) for i, v in acc.items() if v != 0.0]
            items.sort(key=lambda t: t[0])

            for i, v in items:
                indices_out.append(i)
                data_out.append(v)

            indptr_out.append(len(data_out))

        return CSCMatrix(data_out, indices_out, indptr_out, (n_rows, n_cols_out))

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'CSCMatrix':
        """Создание CSC из плотной матрицы."""
        n_rows = len(dense_matrix)
        if n_rows == 0:
            return cls([], [], [0], (0, 0))

        n_cols = len(dense_matrix[0])
        if any(len(row) != n_cols for row in dense_matrix):
            raise ValueError("Плотная матрица должна быть прямоугольной")

        if n_cols == 0:
            return cls([], [], [0], (n_rows, 0))

        data: CSCData = []
        indices: CSCIndices = []
        indptr: CSCIndptr = [0]

        for j in range(n_cols):
            for i in range(n_rows):
                v = float(dense_matrix[i][j])
                if v != 0.0:
                    data.append(v)
                    indices.append(i)
            indptr.append(len(data))

        return cls(data, indices, indptr, (n_rows, n_cols))

    def _to_csr(self) -> 'CSRMatrix':
        from CSR import CSRMatrix 

        n_rows, n_cols = self.shape
        nnz = len(self.data)

        if n_rows == 0:
            return CSRMatrix([], [], [0], (0, n_cols))
        if n_cols == 0:
            return CSRMatrix([], [], [0] * (n_rows + 1), (n_rows, 0))
        if nnz == 0:
            return CSRMatrix([], [], [0] * (n_rows + 1), (n_rows, n_cols))

        row_counts = [0] * n_rows
        for i in self.indices:
            row_counts[i] += 1

        indptr_out: list[int] = [0] * (n_rows + 1)
        for i in range(n_rows):
            indptr_out[i + 1] = indptr_out[i] + row_counts[i]

        next_pos = indptr_out[:] 
        data_out: list[float] = [0.0] * nnz
        indices_out: list[int] = [0] * nnz 

        for j in range(n_cols):
            start, end = self.indptr[j], self.indptr[j + 1]
            for p in range(start, end):
                i = self.indices[p]
                pos = next_pos[i]
                data_out[pos] = float(self.data[p])
                indices_out[pos] = j
                next_pos[i] += 1

        return CSRMatrix(data_out, indices_out, indptr_out, (n_rows, n_cols))

    def _to_coo(self) -> 'COOMatrix':
        from COO import COOMatrix

        n_rows, n_cols = self.shape

        data_out = [float(v) for v in self.data]
        row_out: list[int] = list(self.indices)
        col_out: list[int] = []

        for j in range(n_cols):
            start, end = self.indptr[j], self.indptr[j + 1]
            col_out.extend([j] * (end - start))

        return COOMatrix(data_out, row_out, col_out, (n_rows, n_cols))
