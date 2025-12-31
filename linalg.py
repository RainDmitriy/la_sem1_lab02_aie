from CSC import CSCMatrix
from CSR import CSRMatrix
from types import Vector
from typing import Tuple, Optional

def left_lu_decomposition(A: CSCMatrix) -> Optional[Tuple[CSCMatrix, CSCMatrix]]:
    """
    LU-разложение left-looking для CSC матрицы.
    Возвращает (L, U) - нижнюю и верхнюю треугольные матрицы.
    """
    pass

def right_lu_decomposition(A: CSRMatrix) -> Optional[Tuple[CSRMatrix, CSRMatrix]]:
    """
    LU-разложение right-looking для CSR матрицы.
    Возвращает (L, U) - нижнюю и верхнюю треугольные матрицы.
    """
    pass

def solve_SLAE_left_lu(A: CSCMatrix, b: Vector) -> Optional[Vector]:
    """
    Решение СЛАУ Ax = b через left-looking LU-разложение.
    """
    pass

def solve_SLAE_right_lu(A: CSRMatrix, b: Vector) -> Optional[Vector]:
    """
    Решение СЛАУ Ax = b через right-looking LU-разложение.
    """
    pass

def find_det_with_left_lu(A: CSCMatrix) -> Optional[float]:
    """
    Нахождение определителя через left-looking LU-разложение.
    det(A) = det(L) * det(U)
    """
    pass

def find_det_with_right_lu(A: CSRMatrix) -> Optional[float]:
    """
    Нахождение определителя через right-looking LU-разложение.
    det(A) = det(L) * det(U)
    """
    pass