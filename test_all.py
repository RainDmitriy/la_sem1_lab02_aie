import sys
sys.path.append('.')
from COO import COOMatrix
from CSR import CSRMatrix
from CSC import CSCMatrix

def test_basic():
    print("тестирование базовых операций")
    dense = [
        [0, 0, 3, 4],
        [0, 0, 5, 0],
        [0, 6, 7, 0]
    ]
    print("исходная матрица:")
    for row in dense:
        print(row)

    # COO
    coo = COOMatrix.from_dense(dense)
    print(f"\nCOO: data={coo.data}, row={coo.row}, col={coo.col}")

    # CSR
    csr = CSRMatrix.from_dense(dense)
    print(f"CSR: data={csr.data}, indices={csr.indices}, indptr={csr.indptr}")

    # CSC
    csc = CSCMatrix.from_dense(dense)
    print(f"CSC: data={csc.data}, indices={csc.indices}, indptr={csc.indptr}")

    # Проверка обратного преобразования
    print(f"\nпроверка to_dense():")
    print("COO верно?", coo.to_dense() == dense)
    print("CSR верно?", csr.to_dense() == dense)
    print("CSC верно?", csc.to_dense() == dense)

    # Сложение
    print(f"\nсложение матриц:")
    result = coo + coo
    print("COO + COO:", result.to_dense())

    # Умножение на скаляр
    print(f"\nумножение на скаляр:")
    result = csr * 2
    print("CSR * 2:", result.to_dense())

    # Транспонирование
    print(f"\nтранспонирование:")
    print("CSC транспонирование:", csc.transpose().to_dense())

    # Преобразование форматов
    print(f"\nПреобразование форматов:")
    print("CSR -> COO верно?", csr._to_coo().to_dense() == dense)
    print("CSC -> CSR верно?", csc._to_csr().to_dense() == dense)
    print("COO -> CSC верно?", coo._to_csc().to_dense() == dense)

def test_linalg():
    print("\n\nтестирование линейной алгебры...")
    from linalg import lu_decomposition, solve_SLAE_lu, find_det_with_lu
    A_dense = [
        [2, 1],
        [1, 2]
    ]
    b = [5, 4]
    A = CSCMatrix.from_dense(A_dense)
    print(f"матрица A: {A_dense}")
    print(f"вектор b: {b}")
    lu = lu_decomposition(A)
    if lu:
        L, U = lu
        print("\nLU разложение:")
        print("L:", L.to_dense())
        print("U:", U.to_dense())

    # Решение СЛАУ
    x = solve_SLAE_lu(A, b)
    if x:
        print(f"\nрешение СЛАУ: x = {x}")
        # Проверка: Ax ≈ b
        result = [0, 0]
        for i in range(2):
            for j in range(2):
                result[i] += A_dense[i][j] * x[j]
        print(f"Ax = {result}, ожидалось {b}")

    # Определитель
    det = find_det_with_lu(A)
    if det:
        print(f"\nопределитель A: {det}")
        print(f"ожидаемый: 3 (2*2 - 1*1)")


if __name__ == "__main__":
    test_basic()
    test_linalg()
    print("\nвсе тесты завершены")