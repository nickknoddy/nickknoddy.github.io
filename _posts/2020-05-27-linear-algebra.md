# Matrices and Vectors

- Matrices

  - It is a multi-dimensional array
  - Dimension of matrix = number of rows x number of columns

  [Matrix.png](https://drive.google.com/file/d/1YTcvkYZrbinkvjJ8gWXVAg0YRR6LAKDL/view?usp=drivesdk)

- Elements Of Matrices

  - Elements of the matrix can be accessed using the i or j the term
  - i th term = row && j th term = column

  ![Elements of Matrix.png](/assets/images/linear-algebra/elements-of-matrix.png)

- Vectors
  - It a a matrix with on column and 'n' rows.
    [Vectors.png](https://drive.google.com/file/d/1w7qNXNp52xG_T1tP__-xSzzDaqqj5Dm0/view?usp=drivesdk)
- In general, all our vectors and matrices will be 1-indexed. Note that for some programming languages, the arrays are 0-indexed.
- Matrices are usually denoted by uppercase names while vectors are lowercase.
- "Scalar" means that an object is a single value, not a vector or matrix.

- Octave/Matlab Snippet

  - Code for matrices and vectors

  ```matlab
  % The ; denotes we are going back to a new row.
  A = [1, 2, 3; 4, 5, 6; 7, 8, 9; 10, 11, 12]

  % Initialize a vector
  v = [1;2;3]

  % Get the dimension of the matrix A where m = rows and n = columns
  [m,n] = size(A)

  % You could also store it this way
  dim_A = size(A)

  % Get the dimension of the vector v
  dim_v = size(v)

  % Now let's index into the 2nd row 3rd column of matrix A
  A_23 = A(2,3)
  ```

### Addition and Scalar Multiplication

- Matrix Addition || Subtraction

  - Only matrix with same dimensions can be operated.
  - Size of the output matrix would be same as the input matrix

  [Matrix Addition.png](https://drive.google.com/file/d/1DHGTTSRkK-04xsFoVkLSnOQm-KigHYRp/view?usp=drivesdk)

- Scalar Multiplication || Division

  - Scalar gets multiplied or divided to each and every element of the matrix

  [Scalar Multiplication.png](https://drive.google.com/file/d/1N-3K92gPtMZdLViE1A9KJOQMgbRdCyym/view?usp=drivesdk)

- Combination of Operands
  - Many operations can be combined
    [Combination of operands.png](https://drive.google.com/file/d/1KYwUnN7B_eMrKv_IuJ6i9PVkl648NB5l/view?usp=drivesdk)
- Octave/Matlab Snippet

  - Octave/Matlab commands for matrix addition and scalar multiplication.

  ```matlab
  % Initialize matrix A and B
  A = [1, 2, 4; 5, 3, 2]
  B = [1, 3, 4; 1, 1, 1]

  % Initialize constant s
  s = 2

  % See how element-wise addition works
  add_AB = A + B

  % See how element-wise subtraction works
  sub_AB = A - B

  % See how scalar multiplication works
  mult_As = A * s

  % Divide A by s
  div_As = A / s

  % What happens if we have a Matrix + scalar?
  add_As = A + s
  ```

---

### Matrix Vector Multiplication

- Example 1

  - The result is a vector. The number of columns of the matrix must equal the number of rows of the vector.

  [Example 1.png](https://drive.google.com/file/d/1BlR8NJ8_VqqWMLL3p1Ij9bWh_DCG9INW/view?usp=drivesdk)

- Details

  - An m x n matrix multiplied by an n x 1 vector results in an m x 1 vector.

  [Details.png](https://drive.google.com/file/d/1lKeEg458AQWzQnkPClRj4ZMyFmf1qlp7/view?usp=drivesdk)

- Example 2

  [Example 2.png](https://drive.google.com/file/d/1_tv2s8eHYUOIhTapz8LDDYsKjnKv6V0r/view?usp=drivesdk)

- Hypothesis Trick

  - Hypothesis can be applied to many values efficiently by using matrix-vector multiplication
  - Matrix would be the data set or the value on which hypothesis is to be applied
  - Vector would be the hypothesis
  - Output would be the predictions from the hypothesis

  [Hypothesis trick.png](https://drive.google.com/file/d/1XoVwLS9XvfBbOtCSQYsJqShiohXKbEnW/view?usp=drivesdk)

- Octave/Matlab Snippet

  - matrix-vector multiplication

  ```matlab
  % Initialize matrix A
  A = [1, 2, 3; 4, 5, 6;7, 8, 9]

  % Initialize vector v
  v = [1; 1; 1]

  % Multiply A * v
  Av = A * v
  ```

---

### Matrix Matrix Multiplication

- Example 1

  - multiply two matrices by breaking it into several vector multiplications and concatenating the result.

  [Ex 1.png](https://drive.google.com/file/d/1yQcFaaPbUQ23yjMnQ7BspKkHxTqByv53/view?usp=drivesdk)

- Details

  - An m x n matrix multiplied by an n x o matrix results in an m x o matrix.
  - To multiply two matrices, the number of columns of the first matrix must equal the number of rows of the second matrix.

  [Detail.png](https://drive.google.com/file/d/1xBIVV0JeNYFmp6GR9wJBZdgaaf7X5_JH/view?usp=drivesdk)

- Example 2

  [Ex 2.png](https://drive.google.com/file/d/1U6bb0MRNhyj4c3F3hb-dxfoWH5ZELiE0/view?usp=drivesdk)

- Hypothesis Trick
  - Multiple hypothesis can be calculated using matrix matrix multiplication
    [Hypo Trick.png](https://drive.google.com/file/d/1DUM0sUK6DbvHQsjFqVneHdKIkdcvNEfv/view?usp=drivesdk)
- Octave/Matlab Snippets

  - matrix-matrix multiplication

  ```matlab
  % Initialize a 3 by 2 matrix
  A = [1, 2; 3, 4;5, 6]

  % Initialize a 2 by 1 matrix
  B = [1; 2]

  % We expect a resulting matrix of (3 by 2)*(2 by 1) = (3 by 1)
  mult_AB = A*B

  % Make sure you understand why we got that result
  ```

---

### Matrix Multiplication Properties
https://drive.google.com/file/d/16EGeBuG7VKLiBL7sD-dcGqsMPCSNPRRf/view?usp=drivesdk
- Commutative

  - Matrices are not commutative
    - A _ B ≠ B _ A

  [Commutative.png](https://drive.google.com/file/d/18LsKnrTQ-TUSA8WT-78oLb6hheluVYGv/view?usp=drivesdk)

- Associative

  - Matrices are associative
    - (A _ B) _ C = A _ (B _ C)

  [Associative.png](https://drive.google.com/file/d/1dxl4weqjAYKC_7T9oHYRpos6R4OIbub6/view?usp=drivesdk)

- Identity Matrix

  - The identity matrix, when multiplied by any matrix of the same dimensions, results in the original matrix.
  - When multiplying the identity matrix after some matrix (A∗I), the square identity matrix's dimension should match the other matrix's columns.
  - When multiplying the identity matrix before some other matrix (I∗A), the square identity matrix's dimension should match the other matrix's rows.

  [Identity Matrix.png](https://drive.google.com/file/d/1I9LDkDC7IPIXKhR8k5_JNqoH_f0kzSqc/view?usp=drivesdk)

- ## Octave/Matlab Snippet

  ```matlab
  % Initialize random matrices A and B
  A = [1,2;4,5]
  B = [1,1;0,2]

  % Initialize a 2 by 2 identity matrix
  I = eye(2)

  % The above notation is the same as I = [1,0;0,1]

  % What happens when we multiply I*A ?
  IA = I*A

  % How about A*I ?
  AI = A*I

  % Compute A*B
  AB = A*B

  % Is it equal to B*A?
  BA = B*A

  % Note that IA = AI but AB != BA
  ```

---

### Inverse and Transpose

- Inverse

  - Multiplying by the inverse results in the identity matrix.
  - A non square matrix does not have an inverse matrix.
  - Matrices that don't have an inverse are singular or degenerate.

  [Inverse.png](https://drive.google.com/file/d/1rZEzI5sjmkz4eqXtVwC6UF_ekcmR7Sxt/view?usp=drivesdk)

- Transpose

  - The transposition of a matrix is like rotating the matrix 90° in clockwise direction and then reversing it.

  [Transpose.png](https://drive.google.com/file/d/10Zs9neiSIYFiFR2HZ0K_nQAgjxE4QwAt/view?usp=drivesdk)

- Octave/Matlab Snippet

  - compute inverses of matrices in octave with the pinv(A) function and in Matlab with the inv(A) function.
  - We can compute transposition of matrices in matlab with the transpose(A) function

  ```matlab
  % Initialize matrix A
  A = [1,2,0;0,5,6;7,0,9]

  % Transpose A
  A_trans = A'

  % Take the inverse of A
  A_inv = inv(A)

  % What is A^(-1)*A?
  A_invA = inv(A)*A
  ```

---
