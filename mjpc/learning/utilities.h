#ifndef MJPC_LEARNING_UTILITIES_H_
#define MJPC_LEARNING_UTILITIES_H_

#include <bitset>
#include <cstdlib>
#include <cmath>

// res = 0
template<typename T>
void Zero(T* res, int n) {
  if (n>0) {
    std::memset(res, 0, n*sizeof(T));
  }
}

// res = val
template<typename T>
void Fill(T* res, T val, int n) {
  for (int i=0; i<n; i++) {
    res[i] = val;
  }
}

// res = vec
template<typename T>
void Copy(T* res, const T* vec, int n) {
  if (n>0) {
    std::memcpy(res, vec, n*sizeof(T));
  }
}

// sum(vec)
template<typename T>
T Sum(const T* vec, int n) {
  T res = 0;

  for (int i=0; i<n; i++) {
    res += vec[i];
  }

  return res;
}

// sum(abs(vec))
template<typename T>
T L1(const T* vec, int n) {
  T res = 0;

  for (int i=0; i<n; i++) {
    res += std::abs(vec[i]);
  }

  return res;
}

// res = vec*scl
template<typename T>
void Scale(T* res, const T* vec, T scl, int n) {
  int i = 0;
  for (; i<n; i++) {
    res[i] = vec[i]*scl;
  }
}

// res = vec1 + vec2
template<typename T>
void Add(T* res, const T* vec1, const T* vec2, int n) {
  int i = 0;
  for (; i<n; i++) {
    res[i] = vec1[i] + vec2[i];
  }
}

// res = vec1 - vec2
template<typename T>
void Sub(T* res, const T* vec1, const T* vec2, int n) {
  int i = 0;
  for (; i<n; i++) {
    res[i] = vec1[i] - vec2[i];
  }
}

// res += vec
template<typename T>
void AddTo(T* res, const T* vec, int n) {
  int i = 0;
  for (; i<n; i++) {
    res[i] += vec[i];
  }
}

// res -= vec
template<typename T>
void SubFrom(T* res, const T* vec, int n) {
  int i = 0;
  for (; i<n; i++) {
    res[i] -= vec[i];
  }
}

// res += vec*scale
template<typename T>
void AddToScale(T* res, const T* vec, T scl, int n) {
  int i = 0;
  for (; i<n; i++) {
    res[i] += vec[i]*scl;
  }
}

// res = vec1 + vec2*scl
template<typename T>
void AddScale(T* res, const T* vec1, const T* vec2, T scl, int n) {
  int i = 0;
  for (; i<n; i++) {
    res[i] = vec1[i] + vec2[i]*scl;
  }
}

// vector dot-product
template<typename T>
T Dot(const T* vec1, const T* vec2, int n) {
  T res = 0;
  int i = 0;
  int n_4 = n - 4;

  // do the same order of Additions as the AVX intrinsics implementation.
  // this is faster than the simple for loop you'd expect for a Dot product,
  // and produces exactly the same results.
  T res0 = 0;
  T res1 = 0;
  T res2 = 0;
  T res3 = 0;

  for (; i<=n_4; i+=4) {
    res0 += vec1[i] * vec2[i];
    res1 += vec1[i+1] * vec2[i+1];
    res2 += vec1[i+2] * vec2[i+2];
    res3 += vec1[i+3] * vec2[i+3];
  }
  res = (res0 + res2) + (res1 + res3);

  // process remaining
  int n_i = n - i;
  if (n_i==3) {
    res += vec1[i]*vec2[i] + vec1[i+1]*vec2[i+1] + vec1[i+2]*vec2[i+2];
  } else if (n_i==2) {
    res += vec1[i]*vec2[i] + vec1[i+1]*vec2[i+1];
  } else if (n_i==1) {
    res += vec1[i]*vec2[i];
  }
  return res;
}

// compute vector length (without normalizing)
template<typename T>
T Norm(const T* res, int n) {
  return std::sqrt(Dot(res, res, n));
}

// normalize vector, return length before normalization
template<typename T>
T Normalize(T* res, int n) {
  T norm = (T)std::sqrt(Dot(res, res, n));
  T normInv;

  if (norm<1.0e-15) {
    res[0] = 1;
    for (int i=1; i<n; i++) {
      res[i] = 0;
    }
  } else {
    normInv = 1/norm;
    for (int i=0; i<n; i++) {
      res[i] *= normInv;
    }
  }

  return norm;
}

//------------------------------ matrix-vector operations ------------------------------------------

// multiply matrix and vector
template<typename T>
void MultiplyMatVec(T* res, const T* mat, const T* vec, int nr, int nc) {
  for (int r=0; r<nr; r++) {
    res[r] = Dot(mat + r*nc, vec, nc);
  }
}

// multiply transposed matrix and vector
template<typename T>
void MultiplyMatTVec(T* res, const T* mat, const T* vec, int nr, int nc) {
  T tmp;
  Zero(res, nc);

  for (int r=0; r<nr; r++) {
    if ((tmp = vec[r])) {
      AddToScale(res, mat+r*nc, tmp, nc);
    }
  }
}

// multiply square matrix with vectors on both sides: return vec1'*mat*vec2
template<typename T>
T MultiplyVecMatVec(const T* vec1, const T* mat, const T* vec2, int n) {
  T res = 0;
  for (int i=0; i<n; i++) {
    res += vec1[i] * Dot(mat + i*n, vec2, n);
  }
  return res;
}

//------------------------------ matrix operations -------------------------------------------------

// transpose matrix
template<typename T>
void Transpose(T* res, const T* mat, int nr, int nc) {
  for (int i=0; i<nr; i++) {
    for (int j=0; j<nc; j++) {
      res[j*nr+i] = mat[i*nc+j];
    }
  }
}

// symmetrize square matrix res = (mat + mat')/2
template<typename T>
void Symmetrize(T* res, const T* mat, int n) {
  for (int i=0; i<n; i++) {
    res[i*(n+1)] = mat[i*(n+1)];
    for (int j=0; j<i; j++) {
      res[i*n+j] = res[j*n+i] = 0.5 * (mat[i*n+j] + mat[j*n+i]);
    }
  }
}

// identity matrix
template<typename T>
void Eye(T* mat, int n) {
  Zero(mat, n*n);
  for (int i=0; i<n; i++) {
    mat[i*(n + 1)] = 1;
  }
}

//------------------------------ matrix-matrix operations ------------------------------------------

// multiply matrices, exploit sparsity of mat1
template<typename T>
void MultiplyMatMat(T* res, const T* mat1, const T* mat2,
                   int r1, int c1, int c2) {
  T tmp;

  Zero(res, r1*c2);

  for (int i=0; i<r1; i++) {
    for (int k=0; k<c1; k++) {
      if ((tmp = mat1[i*c1+k])) {
        AddToScale(res+i*c2, mat2+k*c2, tmp, c2);
      }
    }
  }
}

// multiply matrices, second argument transposed
template<typename T>
void MultiplyMatMatT(T* res, const T* mat1, const T* mat2,
                    int r1, int c1, int r2) {
  for (int i=0; i<r1; i++) {
    for (int j=0; j<r2; j++) {
      res[i*r2+j] = Dot(mat1+i*c1, mat2+j*c1, c1);
    }
  }
}

// multiply matrices, first argument transposed
template<typename T>
void MultiplyMatTMat(T* res, const T* mat1, const T* mat2,
                    int r1, int c1, int c2) {
  T tmp;

  Zero(res, c1*c2);

  for (int i=0; i<r1; i++) {
    for (int j=0; j<c1; j++) {
      if ((tmp = mat1[i*c1+j])) {
        AddToScale(res+j*c2, mat2+i*c2, tmp, c2);
      }
    }
  }
}

#endif  // MJPC_LEARNING_UTILITIES_H_
