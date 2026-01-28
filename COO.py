from base import Matrix
from type import COOData, COORows, COOCols, Shape, DenseMatrix


class COOMatrix(Matrix):
    def __init__(self, data: COOData, row: COORows, col: COOCols, shape: Shape):
        
        if not (isinstance(shape, tuple) and len(shape) == 2):
            raise TypeError("shape должен быть кортежем")
        r, c = shape
        if r < 0 or c < 0:
            raise ValueError("Размерности shape должны быть неотрицательными")

        if len(data) != len(row) or len(data) != len(col):
            raise ValueError("data, row и col должны иметь одинаковую длину")
        
        super().__init__(shape)

        self.data: COOData = list(data)
        self.row: COORows = list(row)
        self.col: COOCols = list(col)

        for idx, (ri, ci) in enumerate(zip(self.row, self.col)):
            if not isinstance(ri, int) or not isinstance(ci, int):
                raise TypeError("Индексы row/col должны быть целыми числами")
            if ri < 0 or ri >= r:
                raise IndexError(f"row[{idx}]={ri} вне диапазона [0, {r})")
            if ci < 0 or ci >= c:
                raise IndexError(f"col[{idx}]={ci} вне диапазона [0, {c})")
            
            
    def to_dense(self) -> DenseMatrix:
        r, c = self.shape

        if r == 0:
            return []
        if c == 0:
            return [[] for _ in range(r)]

        dense: DenseMatrix = [[0.0] * c for _ in range(r)]
        for v, i, j in zip(self.data, self.row, self.col):
            dense[i][j] += float(v)
        return dense
    
    
    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        r, c = self.shape

        acc: dict[tuple[int, int], float] = {}
        for v, i, j in zip(self.data, self.row, self.col):
            fv = float(v)
            if fv != 0.0:
                key = (i, j)
                acc[key] = acc.get(key, 0.0) + fv

        if isinstance(other, COOMatrix):
            for v, i, j in zip(other.data, other.row, other.col):
                fv = float(v)
                if fv != 0.0:
                    key = (i, j)
                    acc[key] = acc.get(key, 0.0) + fv
        else:
            b = other.to_dense()
            for i in range(r):
                bi = b[i]
                for j in range(c):
                    fv = float(bi[j])
                    if fv != 0.0:
                        key = (i, j)
                        acc[key] = acc.get(key, 0.0) + fv

        items = [(i, j, v) for (i, j), v in acc.items() if v != 0.0]
        items.sort(key=lambda t: (t[0], t[1]))

        data: COOData = [v for _, _, v in items]
        row: COORows = [i for i, _, _ in items]
        col: COOCols = [j for _, j, _ in items]
        return COOMatrix(data, row, col, (r, c))
    

    def _mul_impl(self, scalar: float) -> 'Matrix':
        r, c = self.shape
        scalar = float(scalar)

        data_out: COOData = [float(v) * scalar for v in self.data]
        return COOMatrix(data_out, list(self.row), list(self.col), (r, c))
    

    def transpose(self) -> 'Matrix':
        r, c = self.shape

        data = [float(v) for v in self.data]
        row = list(self.col)
        col = list(self.row)

        items = list(zip(data, row, col))
        items.sort(key=lambda t: (t[1], t[2]))

        data_t: COOData = [v for v, _, _ in items]
        row_t: COORows = [i for _, i, _ in items]
        col_t: COOCols = [j for _, _, j in items]
        return COOMatrix(data_t, row_t, col_t, (c, r))
    

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        r, k = self.shape
        _, c = other.shape

        if r == 0:
            return COOMatrix([], [], [], (0, c))
        if c == 0:
            return COOMatrix([], [], [], (r, 0))
        if k == 0 or len(self.data) == 0:
            return COOMatrix([], [], [], (r, c))

        acc: dict[tuple[int, int], float] = {}

        if isinstance(other, COOMatrix):
            b_rows: dict[int, list[tuple[int, float]]] = {}
            for bv, bi, bj in zip(other.data, other.row, other.col):
                fb = float(bv)
                if fb == 0.0:
                    continue
                b_rows.setdefault(bi, []).append((bj, fb))

            for av, ai, at in zip(self.data, self.row, self.col):
                fa = float(av)
                if fa == 0.0:
                    continue
                for bj, fb in b_rows.get(at, ()):
                    key = (ai, bj)
                    acc[key] = acc.get(key, 0.0) + fa * fb
        else:
            b = other.to_dense()
            for av, ai, at in zip(self.data, self.row, self.col):
                fa = float(av)
                if fa == 0.0:
                    continue
                bt = b[at]
                for j in range(c):
                    fb = float(bt[j])
                    if fb != 0.0:
                        key = (ai, j)
                        acc[key] = acc.get(key, 0.0) + fa * fb

        items = [(i, j, v) for (i, j), v in acc.items() if v != 0.0]
        items.sort(key=lambda t: (t[0], t[1]))

        data: COOData = [v for _, _, v in items]
        row: COORows = [i for i, _, _ in items]
        col: COOCols = [j for _, j, _ in items]
        return COOMatrix(data, row, col, (r, c))
    

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'COOMatrix':
        r = len(dense_matrix)
        if r == 0:
            return cls([], [], [], (0, 0))

        c = len(dense_matrix[0])
        if any(len(row) != c for row in dense_matrix):
            raise ValueError("Плотная матрица должна быть прямоугольной")

        data: COOData = []
        rows: COORows = []
        cols: COOCols = []

        for i in range(r):
            for j in range(c):
                v = float(dense_matrix[i][j])
                if v != 0.0:
                    data.append(v)
                    rows.append(i)
                    cols.append(j)

        return cls(data, rows, cols, (r, c))
    
    
    def _to_csc(self) -> 'CSCMatrix':
        from CSC import CSCMatrix

        r, c = self.shape

        items = list(zip(self.col, self.row, self.data)) 
        items.sort(key=lambda t: (t[0], t[1]))            

        data_out = [float(v) for _, _, v in items]
        row_ind  = [i for _, i, _ in items]

        col_ptr = [0] * (c + 1)
        for j, _, _ in items:
            col_ptr[j + 1] += 1
        for j in range(c):
            col_ptr[j + 1] += col_ptr[j]

        return CSCMatrix(data_out, row_ind, col_ptr, (r, c))

    def _to_csr(self) -> 'CSRMatrix':
        
        from CSR import CSRMatrix

        r, c = self.shape

        items = list(zip(self.row, self.col, self.data))
        items.sort(key=lambda t: (t[0], t[1]))

        data_out = [float(v) for _, _, v in items]
        col_ind  = [j for _, j, _ in items]

        row_ptr = [0] * (r + 1)
        for i, _, _ in items:
            row_ptr[i + 1] += 1
        for i in range(r):
            row_ptr[i + 1] += row_ptr[i]

        return CSRMatrix(data_out, col_ind, row_ptr, (r, c))