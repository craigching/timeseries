package lm

import (
	"gonum.org/v1/gonum/mat"
)

type LinearModel struct {
	Coeff []float64
}

func New() *LinearModel {
	return &LinearModel{}
}

func (m *LinearModel) Fit(X [][]float64, y []float64) {
	yMat := AsMatrix(y)
	xMat := AsMatrix(X)
	m.Coeff = LinearRegression(xMat, yMat)
}

func (m *LinearModel) Predict(X [][]float64) []float64 {
	yHat := mat.Dense{}
	xMat := AsMatrix(X)
	b := AsMatrix(m.Coeff)

	yHat.Mul(xMat, b)

	return yHat.RawMatrix().Data
}

func LinearRegression(X, y *mat.Dense) []float64 {

	qr := &mat.QR{}
	qr.Factorize(X)

	Q := &mat.Dense{}
	qr.QTo(Q)

	R := &mat.Dense{}
	qr.RTo(R)

	qty := &mat.Dense{}
	qty.Mul(Q.T(), y)

	nCols := X.RawMatrix().Cols - 1

	c := make([]float64, nCols+1)
	for i := nCols; i >= 0; i-- {
		c[i] = qty.At(i, 0)
		for j := i + 1; j < nCols+1; j++ {
			c[i] -= c[j] * R.At(i, j)
		}
		c[i] /= R.At(i, i)
	}

	return c
}

func AsMatrix(A interface{}) *mat.Dense {
	switch A := A.(type) {
	case [][]float64:
		return CnAsMatrix(A)
	case []float64:
		return C1AsMatrix(A)
	}
	return nil
}

func CnAsMatrix(A [][]float64) *mat.Dense {
	nCols := len(A)
	nRows := len(A[0])
	matrix := mat.NewDense(nRows, nCols+1, nil)

	for i := 0; i < nRows; i++ {
		for j := 0; j < nCols+1; j++ {
			if j == 0 {
				matrix.Set(i, 0, 1)
			} else {
				matrix.Set(i, j, A[j-1][i])
			}
		}
	}
	return matrix
}

func C1AsMatrix(y []float64) *mat.Dense {
	nRows := len(y)
	yMat := mat.NewDense(nRows, 1, nil)

	for i := 0; i < nRows; i++ {
		yMat.Set(i, 0, y[i])
	}
	return yMat
}
