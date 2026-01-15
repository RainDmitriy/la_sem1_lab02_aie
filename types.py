# Основные типы данных
DenseMatrix = list[list[float]]  # Плотная матрица: [[row1], [row2], ...] как в NumPy
Shape = tuple[int, int]  # Размерность: (rows, cols)
Vector = list[float]  # Вектор: [1.0, 2.0, 3.0]

# Для COO
COOData = list[float]      # Ненулевые значения
COORows = list[int]        # Индексы строк
COOCols = list[int]        # Индексы столбцов

# Для CSR
CSRData = list[float]      # Ненулевые значения
CSRIndices = list[int]     # Колонки (CSR)
CSRIndptr = list[int]      # Указатели начала строк (CSR)

# Для CSC
CSCData = list[float]      # Ненулевые значения
CSCIndices = list[int]     # Строки (CSC)
CSCIndptr = list[int]      # Указатели начала колонок (CSC)


# Типы для конструкторов
COOArgs = tuple[COOData, COORows, COOCols, Shape]
CSRArgs = tuple[CSRData, CSRIndices, CSRIndptr, Shape]
CSCArgs = tuple[CSRData, CSRIndices, CSRIndptr, Shape]