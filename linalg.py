from CSR import CSRMatrix
from mytypes import Vector
from typing import Tuple, Optional, Dict
from collections import defaultdict

class LUDecomposition:
    """Класс для выполнения LU разложения с выбором главного элемента."""
    
    def __init__(self, matrix: CSRMatrix):
        self.matrix = matrix
        self.n = matrix.shape[0]
        self.L = None
        self.U = None
        self.pivots = list(range(self.n))
    
    def decompose(self) -> bool:
        """Выполняет LU разложение с частичным выбором."""
        U_data = self.matrix.data.copy()
        U_indices = self.matrix.indices.copy()
        U_indptr = self.matrix.indptr.copy()

        L_data = []
        L_indices = []
        L_indptr = [0] * (self.n + 1)
        
        for i in range(self.n):
            L_data.append(1.0)
            L_indices.append(i)
            L_indptr[i + 1] = i + 1

        U = CSRMatrix(U_data, U_indices, U_indptr, (self.n, self.n))
        L = CSRMatrix(L_data, L_indices, L_indptr, (self.n, self.n))

        def get_row_elements(mat: CSRMatrix, row: int) -> Dict[int, float]:
            start = mat.indptr[row]
            end = mat.indptr[row + 1]
            result = {}
            for idx in range(start, end):
                col = mat.indices[idx]
                result[col] = mat.data[idx]
            return result
        
        def update_row(mat: CSRMatrix, row: int, elements: Dict[int, float]) -> CSRMatrix:
            all_rows = []
            for i in range(self.n):
                if i == row:
                    sorted_items = sorted((col, val) for col, val in elements.items() if abs(val) > 1e-12)
                    all_rows.append(sorted_items)
                else:
                    start = mat.indptr[i]
                    end = mat.indptr[i + 1]
                    row_items = [(mat.indices[idx], mat.data[idx]) for idx in range(start, end)]
                    all_rows.append(row_items)

            new_data, new_indices, new_indptr = [], [], [0]
            for row_items in all_rows:
                for col, val in row_items:
                    new_data.append(val)
                    new_indices.append(col)
                new_indptr.append(len(new_data))
            
            return CSRMatrix(new_data, new_indices, new_indptr, (self.n, self.n))

        for k in range(self.n - 1):
            max_row = k
            max_val = abs(U.get_element(k, k) if hasattr(U, 'get_element') else 0.0)

            row_k_elems = get_row_elements(U, k)
            u_kk = row_k_elems.get(k, 0.0)
            
            if abs(u_kk) < 1e-10:
                for i in range(k + 1, self.n):
                    row_i_elems = get_row_elements(U, i)
                    val = abs(row_i_elems.get(k, 0.0))
                    if val > max_val:
                        max_val = val
                        max_row = i
                
                if max_val < 1e-10:
                    return False

                row_k = get_row_elements(U, k)
                row_max = get_row_elements(U, max_row)
                U = update_row(U, k, row_max)
                U = update_row(U, max_row, row_k)

                if k > 0:
                    for j in range(k):
                        l_kj = L.get_element(k, j) if hasattr(L, 'get_element') else 0.0
                        l_mj = L.get_element(max_row, j) if hasattr(L, 'get_element') else 0.0

                        row_k_l = get_row_elements(L, k)
                        row_max_l = get_row_elements(L, max_row)
                        
                        if abs(l_mj) > 1e-12:
                            row_k_l[j] = l_mj
                        elif j in row_k_l:
                            del row_k_l[j]
                        
                        if abs(l_kj) > 1e-12:
                            row_max_l[j] = l_kj
                        elif j in row_max_l:
                            del row_max_l[j]
                        
                        L = update_row(L, k, row_k_l)
                        L = update_row(L, max_row, row_max_l)

                row_k_elems = get_row_elements(U, k)
                u_kk = row_k_elems.get(k, 0.0)

            for i in range(k + 1, self.n):
                row_i_elems = get_row_elements(U, i)
                u_ik = row_i_elems.get(k, 0.0)
                
                if abs(u_ik) > 1e-12:
                    factor = u_ik / u_kk

                    row_i_l = get_row_elements(L, i)
                    row_i_l[k] = factor
                    L = update_row(L, i, row_i_l)

                    new_row_i = {}

                    for col, val in row_i_elems.items():
                        if col != k:
                            new_row_i[col] = val

                    for col, val_k in row_k_elems.items():
                        if col > k:
                            current = new_row_i.get(col, 0.0)
                            new_val = current - factor * val_k
                            if abs(new_val) > 1e-12:
                                new_row_i[col] = new_val
                            elif col in new_row_i:
                                del new_row_i[col]
                    
                    U = update_row(U, i, new_row_i)
        
        self.L = L
        self.U = U
        return True
    
    def get_result(self) -> Optional[Tuple[CSRMatrix, CSRMatrix]]:
        """Возвращает результат разложения (L, U)."""
        if self.L is None or self.U is None:
            return None
        return self.L, self.U


def lu_decomposition(A: CSRMatrix) -> Optional[Tuple[CSRMatrix, CSRMatrix]]:
    """
    LU-разложение матрицы A = L * U.
    Возвращает (L, U) или None если матрица вырождена.
    """
    decomp = LUDecomposition(A)
    if decomp.decompose():
        return decomp.get_result()
    return None


def solve_SLAE_lu(A: CSRMatrix, b: Vector) -> Optional[Vector]:
    """
    Решает систему линейных уравнений A*x = b через LU разложение.
    """
    lu_result = lu_decomposition(A)
    if lu_result is None:
        return None
    
    L, U = lu_result
    n = A.shape[0]

    y = [0.0] * n
    for i in range(n):
        sum_val = 0.0
        start = L.indptr[i]
        end = L.indptr[i + 1]
        for idx in range(start, end):
            j = L.indices[idx]
            if j < i:
                sum_val += L.data[idx] * y[j]
        y[i] = b[i] - sum_val

    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        sum_val = 0.0
        start = U.indptr[i]
        end = U.indptr[i + 1]

        diag = 0.0
        for idx in range(start, end):
            j = U.indices[idx]
            if j == i:
                diag = U.data[idx]
            elif j > i:
                sum_val += U.data[idx] * x[j]
        
        if abs(diag) < 1e-12:
            return None
        
        x[i] = (y[i] - sum_val) / diag
    
    return x


def find_det_with_lu(A: CSRMatrix) -> Optional[float]:
    """
    Вычисляет определитель матрицы через LU разложение.
    """
    lu_result = lu_decomposition(A)
    if lu_result is None:
        return None
    
    _, U = lu_result
    n = A.shape[0]
    
    det = 1.0
    for i in range(n):
        start = U.indptr[i]
        end = U.indptr[i + 1]
        diag = 0.0
        for idx in range(start, end):
            if U.indices[idx] == i:
                diag = U.data[idx]
                break
        det *= diag
    
    return det