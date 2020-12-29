package lm

import (
	"gonum.org/v1/gonum/mat"
)

type LinearModel struct {
	Coeff []float64
	Resid []float64
}

func New() *LinearModel {
	return &LinearModel{}
}

func (m *LinearModel) Fit(X [][]float64, y []float64) {
	c, resid := LinearRegression(X, y)
	m.Coeff = c
	m.Resid = resid
}

func (m *LinearModel) Predict(n int) []float64 {
	// TODO
	return []float64{}
}

func LinearRegression(X [][]float64, y []float64) ([]float64, []float64) {
	nRows := len(y)
	nCols := len(X)
	yMat := mat.NewDense(nRows, 1, nil)
	xMat := mat.NewDense(nRows, nCols+1, nil)

	for i := 0; i < nRows; i++ {
		yMat.Set(i, 0, y[i])
		for j := 0; j < nCols+1; j++ {
			if j == 0 {
				xMat.Set(i, 0, 1)
			} else {
				xMat.Set(i, j, X[j-1][i])
			}
		}
	}

	qr := &mat.QR{}
	Q := &mat.Dense{}
	R := &mat.Dense{}
	qr.Factorize(xMat)

	qr.QTo(Q)
	qr.RTo(R)

	qty := &mat.Dense{}
	qty.Mul(Q.T(), yMat)

	c := make([]float64, nCols+1)
	for i := nCols; i >= 0; i-- {
		c[i] = qty.At(i, 0)
		for j := i + 1; j < nCols+1; j++ {
			c[i] -= c[j] * R.At(i, j)
		}
		c[i] /= R.At(i, i)
	}

	// Residuals
	resid := make([]float64, nRows)
	for i := 0; i < nRows; i++ {
		yHat := c[0]
		for j := 1; j < len(c); j++ {
			yHat += X[j-1][i] * c[j]
		}
		resid[i] = y[i] - yHat
	}

	return c, resid
}
