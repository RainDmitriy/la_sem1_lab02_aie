from CSC import CSCMatrix
from CSR import CSRMatrix
from COO import COOMatrix
from type import Vector
from typing import Tuple, Optional, List, Dict

EPSILON = 1e-12


def lu_decomposition(A: CSCMatrix) -> Optional[Tuple[CSCMatrix, CSCMatrix, List[int]]]:
    n = A.shape[0]
    
    if n != A.shape[1]:
        raise ValueError("LU-разложение требует квадратную матрицу")
    
    A_csr = A._to_csr()
    
    rows_A = [{} for _ in range(n)]
    for i in range(n):
        rows_A[i] = A_csr.get_row_as_dict(i)
    
    L_cols = [{} for _ in range(n)]
    U_rows = [{} for _ in range(n)]
    
    P = list(range(n))
    active_rows = [{} for _ in range(n)]
    
    for k in range(n):
        row_k = {}
        
        for j, val in rows_A[k].items():
            if j >= k:
                row_k[j] = float(val)
        
        for j, val in active_rows[k].items():
            if j >= k:
                current = row_k.get(j, 0.0)
                row_k[j] = current + float(val)
        
        u_kk = row_k.get(k, 0.0)
        
        if abs(u_kk) < EPSILON:
            for i in range(k + 1, n):
                row_i = {}
                for j, val in rows_A[i].items():
                    if j >= k:
                        row_i[j] = float(val)
                for j, val in active_rows[i].items():
                    if j >= k:
                        current = row_i.get(j, 0.0)
                        row_i[j] = current + float(val)
                
                if k in row_i and abs(row_i[k]) > EPSILON:
                    P[k], P[i] = P[i], P[k]
                    rows_A[k], rows_A[i] = rows_A[i], rows_A[k]
                    active_rows[k], active_rows[i] = active_rows[i], active_rows[k]
                    
                    row_k = {}
                    for j, val in rows_A[k].items():
                        if j >= k:
                            row_k[j] = float(val)
                    for j, val in active_rows[k].items():
                        if j >= k:
                            current = row_k.get(j, 0.0)
                            row_k[j] = current + float(val)
                    
                    u_kk = row_k.get(k, 0.0)
                    break
            
            if abs(u_kk) < EPSILON:
                return None
        
        U_rows[k] = {}
        for j, val in row_k.items():
            if j >= k and abs(val) > EPSILON:
                U_rows[k][j] = val
        
        U_rows[k][k] = u_kk
        L_cols[k][k] = 1.0
        
        for i in range(k + 1, n):
            elem = 0.0
            if k in rows_A[i]:
                elem += float(rows_A[i][k])
            if k in active_rows[i]:
                elem += float(active_rows[i][k])
            
            if abs(elem) > EPSILON:
                L_ik = elem / u_kk
                L_cols[k][i] = L_ik
                
                for j, U_kj in U_rows[k].items():
                    if j > k:
                        delta = -L_ik * U_kj
                        if abs(delta) > EPSILON:
                            current = active_rows[i].get(j, 0.0)
                            active_rows[i][j] = current + delta
    
    L_data, L_indices, L_indptr = [], [], [0]
    for j in range(n):
        rows = sorted(L_cols[j].keys())
        for i in rows:
            L_data.append(L_cols[j][i])
            L_indices.append(i)
        L_indptr.append(len(L_data))
    
    U_cols = [{} for _ in range(n)]
    for i in range(n):
        for j, val in U_rows[i].items():
            U_cols[j][i] = val
    
    U_data, U_indices, U_indptr = [], [], [0]
    for j in range(n):
        rows = sorted(U_cols[j].keys())
        for i in rows:
            U_data.append(U_cols[j][i])
            U_indices.append(i)
        U_indptr.append(len(U_data))
    
    return CSCMatrix(L_data, L_indices, L_indptr, (n, n)), CSCMatrix(U_data, U_indices, U_indptr, (n, n)), P


def solve_SLAE_lu(A: CSCMatrix, b: Vector) -> Optional[Vector]:
    lu_result = lu_decomposition(A)
    
    if lu_result is None:
        return None
    
    L, U, P = lu_result
    n = A.shape[0]
    
    b_perm = [b[P[i]] for i in range(n)]
    
    L_csr = L._to_csr()
    
    y = [0.0] * n
    for i in range(n):
        s = 0.0
        row_dict = L_csr.get_row_as_dict(i)
        for j, val in row_dict.items():
            if j < i:
                s += val * y[j]
        y[i] = b_perm[i] - s
    
    U_csr = U._to_csr()
    
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = 0.0
        row_dict = U_csr.get_row_as_dict(i)
        diag_val = 0.0
        
        for j, val in row_dict.items():
            if j > i:
                s += val * x[j]
            elif j == i:
                diag_val = val
        
        if abs(diag_val) < EPSILON:
            return None
        
        x[i] = (y[i] - s) / diag_val
    
    return x


def find_det_with_lu(A: CSCMatrix) -> Optional[float]:
    lu_result = lu_decomposition(A)
    
    if lu_result is None:
        return None
    
    _, U, P = lu_result
    n = A.shape[0]
    
    visited = [False] * n
    swaps = 0
    
    for i in range(n):
        if not visited[i]:
            cycle_len = 0
            j = i
            while not visited[j]:
                visited[j] = True
                j = P[j]
                cycle_len += 1
            if cycle_len > 1:
                swaps += cycle_len - 1
    
    sign = 1 if swaps % 2 == 0 else -1
    
    det = sign
    for i in range(n):
        diag = U.get(i, i)
        if abs(diag) < EPSILON:
            return 0.0
        det *= diag
    
    return det