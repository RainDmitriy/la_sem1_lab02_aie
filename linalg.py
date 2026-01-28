from CSC import CSCMatrix
from type import Vector
from typing import Tuple, Optional

TOL = 1e-10

def lu_decomposition(A: CSCMatrix) -> Optional[Tuple[CSCMatrix, CSCMatrix]]:
    n = A.shape[0]

    L_cols = [{} for _ in range(n)]
    U_rows = [{} for _ in range(n)]
    
    rows_A = [{} for _ in range(n)]
    for j in range(n):
        for pos in range(A.indptr[j], A.indptr[j + 1]):
            i = A.indices[pos]
            rows_A[i][j] = float(A.data[pos])
    
    active_rows = [{} for _ in range(n)]
    
    for k in range(n):
        cur_row = {}
        
        for j, val in rows_A[k].items():
            if j >= k:
                cur_row[j] = float(val)
        
        for j, val in active_rows[k].items():
            if j >= k:
                cur = cur_row.get(j, 0)
                cur_row[j] = cur + float(val)
        
        diag = cur_row.get(k, 0)
        if abs(diag) < TOL:
            return None
        
        U_rows[k] = {}
        for j, val in cur_row.items():
            if j >= k and abs(val) > TOL:
                U_rows[k][j] = val
        
        U_rows[k][k] = diag
        L_cols[k][k] = 1
        
        for i in range(k + 1, n):
            elem = 0
            if k in rows_A[i]:
                elem += float(rows_A[i][k])
            if k in active_rows[i]:
                elem += float(active_rows[i][k])
            
            if abs(elem) > TOL:
                L_ik = elem / diag
                L_cols[k][i] = L_ik
                
                for j, U_kj in U_rows[k].items():
                    if j > k:
                        delta = -L_ik * U_kj
                        if abs(delta) > TOL:
                            curr = active_rows[i].get(j, 0)
                            active_rows[i][j] = curr + delta
    
    L_data, L_idx, L_ptr = [], [], [0]
    for j in range(n):
        rows_sorted = sorted(L_cols[j].keys())
        for i in rows_sorted:
            L_data.append(L_cols[j][i])
            L_idx.append(i)
        L_ptr.append(len(L_data))
    
    U_cols = [{} for _ in range(n)]
    for i in range(n):
        for j, val in U_rows[i].items():
            U_cols[j][i] = val
    
    U_data, U_idx, U_ptr = [], [], [0]
    for j in range(n):
        rows_sorted = sorted(U_cols[j].keys())
        for i in rows_sorted:
            U_data.append(U_cols[j][i])
            U_idx.append(i)
        U_ptr.append(len(U_data))
    
    return CSCMatrix(L_data, L_idx, L_ptr, (n, n)), CSCMatrix(U_data, U_idx, U_ptr, (n, n))


def solve_SLAE_lu(A: CSCMatrix, b: Vector) -> Optional[Vector]:
    lu_res = lu_decomposition(A)

    if lu_res is None:
        return None
    
    L, U = lu_res
    n = A.shape[0]
    
    L_dense = L.to_dense()
    U_dense = U.to_dense()
    
    y = [0] * n
    for i in range(n):
        s = 0
        for j in range(i):
            s += L_dense[i][j] * y[j]

        y[i] = b[i] - s
    
    x = [0] * n
    for i in range(n - 1, -1, -1):
        s = 0
        for j in range(i + 1, n):
            s += U_dense[i][j] * x[j]
        
        if abs(U_dense[i][i]) < TOL:
            return None
        
        x[i] = (y[i] - s) / U_dense[i][i]
    
    return x

def find_det_with_lu(A: CSCMatrix) -> Optional[float]:
    lu_res = lu_decomposition(A)

    if lu_res is None:
        return None
    
    L, U = lu_res
    n = A.shape[0]
    
    U_dense = U.to_dense()
    
    det_L = 1.0
    
    det_U = 1
    for i in range(n):
        det_U *= U_dense[i][i]
    
    return det_L * det_U