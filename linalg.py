from CSC import CSCMatrix
from type import Vector
from typing import Tuple, Optional

TOL = 1e-10

def lu_decomposition(A_mat: CSCMatrix) -> Optional[Tuple[CSCMatrix, CSCMatrix]]:
    n_val = A_mat.shape[0]

    L_cols_list = [{} for _ in range(n_val)]
    U_rows_list = [{} for _ in range(n_val)]
    
    rows_of_A = [{} for _ in range(n_val)]
    for j_idx in range(n_val):
        for pos in range(A_mat.ptrs[j_idx], A_mat.ptrs[j_idx + 1]):
            i_idx = A_mat.idxs[pos]
            rows_of_A[i_idx][j_idx] = float(A_mat.d[pos])
    
    active_rows_list = [{} for _ in range(n_val)]
    
    for k_idx in range(n_val):
        current_row = {}
        
        for j_idx, val in rows_of_A[k_idx].items():
            if j_idx >= k_idx:
                current_row[j_idx] = float(val)
        
        for j_idx, val in active_rows_list[k_idx].items():
            if j_idx >= k_idx:
                current_val = current_row.get(j_idx, 0)
                current_row[j_idx] = current_val + float(val)
        
        diag_val = current_row.get(k_idx, 0)
        if abs(diag_val) < TOL:
            return None
        
        U_rows_list[k_idx] = {}
        for j_idx, val in current_row.items():
            if j_idx >= k_idx and abs(val) > TOL:
                U_rows_list[k_idx][j_idx] = val
        
        U_rows_list[k_idx][k_idx] = diag_val
        L_cols_list[k_idx][k_idx] = 1
        
        for i_idx in range(k_idx + 1, n_val):
            elem_val = 0
            if k_idx in rows_of_A[i_idx]:
                elem_val += float(rows_of_A[i_idx][k_idx])
            if k_idx in active_rows_list[i_idx]:
                elem_val += float(active_rows_list[i_idx][k_idx])
            
            if abs(elem_val) > TOL:
                L_val = elem_val / diag_val
                L_cols_list[k_idx][i_idx] = L_val
                
                for j_idx, U_val in U_rows_list[k_idx].items():
                    if j_idx > k_idx:
                        delta = -L_val * U_val
                        if abs(delta) > TOL:
                            curr = active_rows_list[i_idx].get(j_idx, 0)
                            active_rows_list[i_idx][j_idx] = curr + delta
    
    L_vals, L_idxs, L_ptrs = [], [], [0]
    for j_idx in range(n_val):
        row_idxs_sorted = sorted(L_cols_list[j_idx].keys())
        for i_idx in row_idxs_sorted:
            L_vals.append(L_cols_list[j_idx][i_idx])
            L_idxs.append(i_idx)
        L_ptrs.append(len(L_vals))
    
    U_cols_list = [{} for _ in range(n_val)]
    for i_idx in range(n_val):
        for j_idx, val in U_rows_list[i_idx].items():
            U_cols_list[j_idx][i_idx] = val
    
    U_vals, U_idxs, U_ptrs = [], [], [0]
    for j_idx in range(n_val):
        row_idxs_sorted = sorted(U_cols_list[j_idx].keys())
        for i_idx in row_idxs_sorted:
            U_vals.append(U_cols_list[j_idx][i_idx])
            U_idxs.append(i_idx)
        U_ptrs.append(len(U_vals))
    
    return CSCMatrix(L_vals, L_idxs, L_ptrs, (n_val, n_val)), CSCMatrix(U_vals, U_idxs, U_ptrs, (n_val, n_val))


def solve_SLAE_lu(A_mat: CSCMatrix, b_vec: Vector) -> Optional[Vector]:
    lu_res = lu_decomposition(A_mat)

    if lu_res is None:
        return None
    
    L_mat, U_mat = lu_res
    n_val = A_mat.shape[0]
    
    L_dense_mat = L_mat.to_dense()
    U_dense_mat = U_mat.to_dense()
    
    y_vec = [0] * n_val
    for i_idx in range(n_val):
        s_val = 0
        for j_idx in range(i_idx):
            s_val += L_dense_mat[i_idx][j_idx] * y_vec[j_idx]

        y_vec[i_idx] = b_vec[i_idx] - s_val
    
    x_vec = [0] * n_val
    for i_idx in range(n_val - 1, -1, -1):
        s_val = 0
        for j_idx in range(i_idx + 1, n_val):
            s_val += U_dense_mat[i_idx][j_idx] * x_vec[j_idx]
        
        if abs(U_dense_mat[i_idx][i_idx]) < TOL:
            return None
        
        x_vec[i_idx] = (y_vec[i_idx] - s_val) / U_dense_mat[i_idx][i_idx]
    
    return x_vec

def find_det_with_lu(A_mat: CSCMatrix) -> Optional[float]:
    lu_res = lu_decomposition(A_mat)

    if lu_res is None:
        return None
    
    L_mat, U_mat = lu_res
    n_val = A_mat.shape[0]
    
    U_dense_mat = U_mat.to_dense()
    
    det_L_val = 1.0
    
    det_U_val = 1
    for i_idx in range(n_val):
        det_U_val *= U_dense_mat[i_idx][i_idx]
    
    return det_L_val * det_U_val