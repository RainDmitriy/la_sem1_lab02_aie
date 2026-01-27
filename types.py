# type.py
from typing import List, Tuple

# Основные типы данных
DenseMatrix = List[List[float]]  # Плотная матрица: [[row1], [row2], ...] как в NumPy
Shape = Tuple[int, int]  # Размерность: (rows, cols)
Vector = List[float]  # Вектор: [1.0, 2.0, 3.0]

# Для COO
COOData = List[float]      # Ненулевые значения
COORows = List[int]        # Индексы строк
COOCols = List[int]        # Индексы столбцов

# Для CSR и CSC
CSRData = CSCData = List[float]      # Ненулевые значения
CSRIndices = CSCIndices = List[int]  # Колонки (CSR) или строки (CSC)
CSRIndptr = CSCIndptr = List[int]    # Указатели начала строк (CSR) или колонок (CSC)

# Типы для конструкторов
COOArgs = Tuple[COOData, COORows, COOCols, Shape]
CSRArgs = Tuple[CSRData, CSRIndices, CSRIndptr, Shape]

CSCArgs = Tuple[CSCData, CSCIndices, CSCIndptr, Shape]
