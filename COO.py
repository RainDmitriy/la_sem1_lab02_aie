from collections import defaultdict
from base import Matrix
from matrix_types import COOData, COORows, COOCols, Shape, DenseMatrix


class COOMatrix(Matrix):
    def __init__(self, data: COOData, row: COORows, col: COOCols, shape: Shape):
        super().__init__(shape)
        self.data = data
        self.row = row
        self.col = col
        self.nnz = len(data)

    def to_dense(self) -> DenseMatrix:
        mat = [[0.0] * self.shape[1] for _ in range(self.shape[0])]
        for r, c, v in zip(self.row, self.col, self.data):
            mat[r][c] += v
        return mat

    def _add_impl(self, other: 'Matrix') -> 'Matrix':
        if not isinstance(other, COOMatrix):
             other = other._to_coo()

        merged = defaultdict(float)
        
        for r, c, v in zip(self.row, self.col, self.data):
            merged[(r, c)] += v
            
        for r, c, v in zip(other.row, other.col, other.data):
            merged[(r, c)] += v
            
        # FIX: Не фильтруем нули, чтобы пройти тесты на структуру
        # Сортируем ключи для детерминизма
        sorted_keys = sorted(merged.keys())
        new_data, new_rows, new_cols = [], [], []
        
        for r, c in sorted_keys:
            new_rows.append(r)
            new_cols.append(c)
            new_data.append(merged[(r, c)])
                
        return COOMatrix(new_data, new_rows, new_cols, self.shape)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        # FIX: Сохраняем структуру даже при умножении на 0
        return COOMatrix([x * scalar for x in self.data], list(self.row), list(self.col), self.shape)

    def transpose(self) -> 'Matrix':
        # FIX: Сортируем результат, чтобы тесты не падали из-за порядка
        triplets = sorted(zip(self.col, self.row, self.data))
        new_rows = [t[0] for t in triplets]
        new_cols = [t[1] for t in triplets]
        new_data = [t[2] for t in triplets]
        return COOMatrix(new_data, new_rows, new_cols, (self.shape[1], self.shape[0]))

    def _matmul_impl(self, other: 'Matrix') -> 'Matrix':
        if not isinstance(other, COOMatrix):
             other = other._to_coo()

        b_lookup = defaultdict(list)
        for r, c, v in zip(other.row, other.col, other.data):
            b_lookup[r].append((c, v))

        result_map = defaultdict(float)
        
        for r_a, c_a, v_a in zip(self.row, self.col, self.data):
            if c_a in b_lookup:
                for c_b, v_b in b_lookup[c_a]:
                    result_map[(r_a, c_b)] += v_a * v_b

        # Сортируем для детерминизма и не удаляем нули
        sorted_keys = sorted(result_map.keys())
        new_data, new_rows, new_cols = [], [], []
        for r, c in sorted_keys:
            new_rows.append(r)
            new_cols.append(c)
            new_data.append(result_map[(r, c)])
                
        return COOMatrix(new_data, new_rows, new_cols, (self.shape[0], other.shape[1]))

    @classmethod
    def from_dense(cls, dense_matrix: DenseMatrix) -> 'COOMatrix':
        if not dense_matrix:
            return cls([], [], [], (0, 0))
        rows_idx, cols_idx, data = [], [], []
        n, m = len(dense_matrix), len(dense_matrix[0])
        
        for i in range(n):
            for j in range(m):
                val = dense_matrix[i][j]
                if val != 0:
                    rows_idx.append(i)
                    cols_idx.append(j)
                    data.append(val)
                    
        return cls(data, rows_idx, cols_idx, (n, m))

    def _to_csc(self) -> 'CSCMatrix':
        from CSC import CSCMatrix
        # FIX: Суммируем дубликаты перед конвертацией
        merged = defaultdict(float)
        for r, c, v in zip(self.row, self.col, self.data):
            merged[(r, c)] += v
            
        # Сортируем по (col, row) для CSC
        sorted_items = sorted(merged.items(), key=lambda x: (x[0][1], x[0][0]))
        
        data = [x[1] for x in sorted_items]
        indices = [x[0][0] for x in sorted_items]
        
        indptr = [0] * (self.shape[1] + 1)
        col_counts = defaultdict(int)
        for (r, c), _ in sorted_items:
            col_counts[c] += 1
            
        cum_sum = 0
        for i in range(self.shape[1]):
            indptr[i] = cum_sum
            cum_sum += col_counts[i]
        indptr[self.shape[1]] = cum_sum
        
        return CSCMatrix(data, indices, indptr, self.shape)

    def _to_csr(self) -> 'CSRMatrix':
        from CSR import CSRMatrix
        # FIX: Суммируем дубликаты
        merged = defaultdict(float)
        for r, c, v in zip(self.row, self.col, self.data):
            merged[(r, c)] += v
            
        # Сортируем по (row, col) для CSR
        sorted_items = sorted(merged.items(), key=lambda x: (x[0][0], x[0][1]))
        
        data = [x[1] for x in sorted_items]
        indices = [x[0][1] for x in sorted_items]
        
        indptr = [0] * (self.shape[0] + 1)
        row_counts = defaultdict(int)
        for (r, c), _ in sorted_items:
            row_counts[r] += 1
            
        cum_sum = 0
        for i in range(self.shape[0]):
            indptr[i] = cum_sum
            cum_sum += row_counts[i]
        indptr[self.shape[0]] = cum_sum
        
        return CSRMatrix(data, indices, indptr, self.shape)