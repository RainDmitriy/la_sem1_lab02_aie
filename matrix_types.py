from typing import Optional, Tuple, List, Union


# Основные типы данных
DenseMatrix = List[List[float]]  # Плотная матрица: [[row1], [row2], ...] как в NumPy
Shape = Tuple[int, int]  # Размерность: (rows, cols)
Vector = List[float]  # Вектор: [1.0, 2.0, 3.0]

# Для COO
COOData = List[float]      # Ненулевые значения
COORows = List[int]        # Индексы строк
COOCols = List[int]        # Индексы столбцов

# Для CSR
CSRData = List[float]      # Ненулевые значения
CSRIndices = List[int]     # Колонки (CSR)
CSRIndptr = List[int]      # Указатели начала строк (CSR)

# Для CSC
CSCData = List[float]      # Ненулевые значения
CSCIndices = List[int]     # Строки (CSC)
CSCIndptr = List[int]      # Указатели начала колонок (CSC)


# Типы для конструкторов
COOArgs = Tuple[COOData, COORows, COOCols, Shape]
CSRArgs = Tuple[CSRData, CSRIndices, CSRIndptr, Shape]
CSCArgs = Tuple[CSRData, CSRIndices, CSRIndptr, Shape]