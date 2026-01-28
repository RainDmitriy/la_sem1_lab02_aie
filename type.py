

from typing import List, Tuple

DenseMatrix = List[List[float]]
Shape = Tuple[int, int]
Vector = List[float]
COOData = List[float]
COORows = List[int]
COOCols = List[int]
CSRData = CSCData = List[float]
CSRIndices = CSCIndices = List[int]
CSRIndptr = CSCIndptr = List[int]
COOArgs = Tuple[COOData, COORows, COOCols, Shape]
CSRArgs = Tuple[CSRData, CSRIndices, CSRIndptr, Shape]
CSCArgs = Tuple[CSCData, CSCIndices, CSCIndptr, Shape]