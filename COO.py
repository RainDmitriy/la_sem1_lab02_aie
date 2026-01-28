from base import Matrix
from typing import Dict, Tuple, List
from type import COOData, COORows, COOCols, Shape, DenseMatrix

class COOMatrix(Matrix):
    def _to_coo(self) -> 'COOMatrix':
        """COO -> COO (просто возвращаем копию)"""
        return COOMatrix(self.data.copy(), self.row.copy(), self.col.copy(), self.shape)

    def _normalize(self) -> None:
        """Нормализует COO матрицу"""
        mp = {}
        for d, r, c in zip(self.data, self.row, self.col):
            d = float(d)
            if d == 0.0:
                continue
            mp[(int(r), int(c))] = mp.get((int(r), int(c)), 0.0) + d
        items = [((r, c), d) for (r, c), d in mp.items() if d != 0.0]
        items.sort(key=lambda x: (x[0][0], x[0][1]))
        self.row = [r for (r, _), _ in items]
        self.col = [c for (_, c), _ in items]
        self.data = [d for _, d in items]
        self.nnz = len(self.data)
        self.row = [r for (r, _), _ in items]
        self.col = [c for (_, c), _ in items]
        self.data = [d for _, d in items]

    def __init__(self, data: COOData, row: COORows, col: COOCols, shape: Shape):
        super().__init__(shape)
        self.data = data
        self.row = row
        self.col = col
        self.nnz = len(data)

        if len(data) != len(row) or len(data) != len(col):
            raise ValueError("data, row и col не совпадают")

        for r, c in zip(row, col):
            if r < 0 or r >= shape[0] or c < 0 or c >= shape[1]:
                raise ValueError("индекс за границой матрицы")
        self._normalize()

    def to_dense(self) -> DenseMatrix:
        """из COO в плотную матрицу"""
        rows, cols = self.shape
        dense = [[0.0] * cols for _ in range(rows)]
        for value, r, c in zip(self.data, self.row, self.col):
            dense[r][c] = value
        return dense

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        """Сложение COO матриц"""
        if hasattr(other, "_to_coo"):
            b = other._to_coo()
        else:
            b = COOMatrix.from_dense(other.to_dense())
        mp: Dict[Tuple[int, int], float] = {}
        for d, r, c in zip(self.data, self.row, self.col):
            mp[(r, c)] = mp.get((r, c), 0.0) + float(d)
        for d, r, c in zip(b.data, b.row, b.col):
            mp[(r, c)] = mp.get((r, c), 0.0) + float(d)
        data, row, col = [], [], []
        for (r, c), d in mp.items():
            if abs(d) > 1e-14:
                data.append(d)
                row.append(r)
                col.append(c)

        return COOMatrix(data, row, col, self.shape)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        """Умножение матрицы на число."""
        data = [float(d) * float(scalar) for d in self.data]
        return COOMatrix(data, self.row[:], self.col[:], self.shape)

    def transpose(self) -> 'Matrix':
        """транспонирование"""
        new_shape = (self.shape[1], self.shape[0])
        return COOMatrix(self.data.copy(), self.col.copy(), self.row.copy(), new_shape)

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        """реализация умножения матриц."""
        if self.shape[1] != other.shape[0]:
            raise ValueError("несовместимые размеры матриц")
        sr, sc = self.shape[0], other.shape[1]
        mp = {}
        if hasattr(other, "_to_csr"):
            b = other._to_csr()
        else:
            b = COOMatrix.from_dense(other.to_dense())._to_csr()

        for p in range(len(self.data)):
            r = self.row[p]
            c = self.col[p]
            da = float(self.data[p])
            s = b.indptr[c]
            e = b.indptr[c + 1]
            for p2 in range(s, e):
                c2 = b.indices[p2]
                db = float(b.data[p2])
                key = (r, c2)
                mp[key] = mp.get(key, 0.0) + da * db
        data, row, col = [], [], []
        for (r, c), d in mp.items():
            if abs(d) > 1e-14:
                data.append(d)
                row.append(r)
                col.append(c)

        return COOMatrix(data, row, col, (sr, sc))

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'COOMatrix':
        """создание coo из плотной матрицы"""
        rows = len(dense_matrix)
        cols = len(dense_matrix[0]) if rows > 0 else 0
        data = []
        row_indices = []
        col_indices = []
        for i in range(rows):
            for j in range(cols):
                value = dense_matrix[i][j]
                if abs(value) > 1e-10:  #ноль = очень маленькие значения
                    data.append(value)
                    row_indices.append(i)
                    col_indices.append(j)

        return cls(data, row_indices, col_indices, (rows, cols))

    def _to_csc(self) -> 'CSCMatrix':
        """преобразование COOMatrix в CSCMatrix"""
        from CSC import CSCMatrix
        #сортируем элементы по столбцам и строкам
        sorted_indices = sorted(range(self.nnz), key=lambda k: (self.col[k], self.row[k]))
        data = []
        indices = []
        indptr = [0] * (self.shape[1] + 1)
        current_col = 0
        for idx in sorted_indices:
            col = self.col[idx]
            while current_col < col:
                indptr[current_col + 1] = len(data)
                current_col += 1
            data.append(self.data[idx])
            indices.append(self.row[idx])
        for col in range(current_col, self.shape[1]):
            indptr[col + 1] = len(data)

        return CSCMatrix(data, indices, indptr, self.shape)

    def _to_csr(self) -> 'CSRMatrix':
        """преобразование COOMatrix в CSRMatrix"""
        from CSR import CSRMatrix
        sorted_indices = sorted(range(self.nnz), key=lambda k: (self.row[k], self.col[k]))
        data = []
        indices = []
        indptr = [0] * (self.shape[0] + 1)
        current_row = 0
        for idx in sorted_indices:
            row = self.row[idx]
            while current_row < row:
                indptr[current_row + 1] = len(data)
                current_row += 1
            data.append(self.data[idx])
            indices.append(self.col[idx])
        for row in range(current_row, self.shape[0]):
            indptr[row + 1] = len(data)

        return CSRMatrix(data, indices, indptr, self.shape)

    def __str__(self) -> str:
        return f"COOMatrix(shape={self.shape}, nnz={self.nnz})"

    def __repr__(self) -> str:
        return self.__str__()