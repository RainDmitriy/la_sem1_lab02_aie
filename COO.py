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
            
        # Сортировка результата для детерминизма (здесь она неизбежна, но NNZ обычно меньше)
        sorted_keys = sorted(merged.keys())
        new_data, new_rows, new_cols = [], [], []
        for r, c in sorted_keys:
            new_rows.append(r)
            new_cols.append(c)
            new_data.append(merged[(r, c)])
                
        return COOMatrix(new_data, new_rows, new_cols, self.shape)

    def _mul_impl(self, scalar: float) -> 'Matrix':
        return COOMatrix([x * scalar for x in self.data], list(self.row), list(self.col), self.shape)

    def transpose(self) -> 'Matrix':
        # Линейная сортировка (Bucket Sort) для транспонирования
        # Нам нужно отсортировать по столбцам (которые станут строками)
        new_shape = (self.shape[1], self.shape[0])
        # Считаем количество элементов в каждой новой строке
        counts = [0] * new_shape[0]
        for c in self.col:
            counts[c] += 1
            
        # Строим позиции
        positions = [0] * new_shape[0]
        cum_sum = 0
        for i in range(new_shape[0]):
            positions[i] = cum_sum
            cum_sum += counts[i]
            
        # Заполняем
        new_rows = [0] * self.nnz
        new_cols = [0] * self.nnz
        new_data = [0.0] * self.nnz
        
        for r, c, v in zip(self.row, self.col, self.data):
            pos = positions[c]
            new_rows[pos] = c
            new_cols[pos] = r
            new_data[pos] = v
            positions[c] += 1
            
        return COOMatrix(new_data, new_rows, new_cols, new_shape)

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
        
        # 1. Суммируем дубликаты (стандарт COO)
        # Используем dict, это самый надежный способ убрать дубли
        summed = defaultdict(float)
        for r, c, v in zip(self.row, self.col, self.data):
            summed[(r, c)] += v
            
        # Если пусто
        if not summed:
            return CSCMatrix([], [], [0] * (self.shape[1] + 1), self.shape)

        # 2. Bucket Sort (Линейная сложность)
        # Считаем сколько элементов в каждом столбце
        col_counts = [0] * self.shape[1]
        for r, c in summed.keys():
            col_counts[c] += 1
            
        indptr = [0] * (self.shape[1] + 1)
        cum = 0
        for i in range(self.shape[1]):
            indptr[i] = cum
            cum += col_counts[i]
        indptr[self.shape[1]] = cum
        
        # Создаем массивы
        nnz_summed = len(summed)
        indices = [0] * nnz_summed
        data = [0.0] * nnz_summed
        
        # Курсоры для заполнения
        cursors = list(indptr)
        
        # Заполняем (порядок внутри столбца пока случайный!)
        for (r, c), v in summed.items():
            pos = cursors[c]
            indices[pos] = r
            data[pos] = v
            cursors[c] += 1
            
        # 3. Сортировка внутри каждого столбца (Обязательно для CSC!)
        # Сортируем маленькие кусочки - это быстро
        for j in range(self.shape[1]):
            start, end = indptr[j], indptr[j+1]
            if end - start > 1:
                # Сортируем срез по индексам строк
                # Zip -> Sort -> Unzip
                part = sorted(zip(indices[start:end], data[start:end]))
                for k, (idx, val) in enumerate(part):
                    indices[start + k] = idx
                    data[start + k] = val
                    
        return CSCMatrix(data, indices, indptr, self.shape)

    def _to_csr(self) -> 'CSRMatrix':
        from CSR import CSRMatrix
        
        summed = defaultdict(float)
        for r, c, v in zip(self.row, self.col, self.data):
            summed[(r, c)] += v
            
        if not summed:
            return CSRMatrix([], [], [0] * (self.shape[0] + 1), self.shape)

        row_counts = [0] * self.shape[0]
        for r, c in summed.keys():
            row_counts[r] += 1
            
        indptr = [0] * (self.shape[0] + 1)
        cum = 0
        for i in range(self.shape[0]):
            indptr[i] = cum
            cum += row_counts[i]
        indptr[self.shape[0]] = cum
        
        indices = [0] * len(summed)
        data = [0.0] * len(summed)
        cursors = list(indptr)
        
        for (r, c), v in summed.items():
            pos = cursors[r]
            indices[pos] = c
            data[pos] = v
            cursors[r] += 1
            
        # Сортировка внутри каждой строки
        for i in range(self.shape[0]):
            start, end = indptr[i], indptr[i+1]
            if end - start > 1:
                part = sorted(zip(indices[start:end], data[start:end]))
                for k, (idx, val) in enumerate(part):
                    indices[start + k] = idx
                    data[start + k] = val
                    
        return CSRMatrix(data, indices, indptr, self.shape)
