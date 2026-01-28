# type.py
from typing import List, Tuple

# Основные типы данных
DenseMatrix = List[List[float]]
Shape = Tuple[int, int]
Vector = List[float]

# Для COO
COOData = List[float]
COORows = List[int]
COOCols = List[int]

# Для CSR и CSC
CSRData = CSCData = List[float]
CSRIndices = CSCIndices = List[int]
CSRIndptr = CSCIndptr = List[int]

# Типы для конструкторов
COOArgs = Tuple[COOData, COORows, COOCols, Shape]
CSRArgs = Tuple[CSRData, CSRIndices, CSRIndptr, Shape]
CSCArgs = Tuple[CSCData, CSCIndices, CSCIndptr, Shape]

