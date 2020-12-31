package ar

import (
	"github.com/craigching/timeseries/lm"
	"gonum.org/v1/gonum/mat"
)

type AutoRegressiveModel struct {
	lm.LinearModel
	order int
	y     []float64
}

func New(order int) *AutoRegressiveModel {
	return &AutoRegressiveModel{
		LinearModel: lm.LinearModel{},
		order:       order,
	}
}

func (m *AutoRegressiveModel) Fit(x []float64) {
	var X [][]float64
	// n is the length of the lag arrays
	n := len(x) - m.order
	// build lags
	for i := 0; i < m.order; i++ {
		X = append(X, lag(x, i+1, n))
	}
	// Save y so we can predict from it
	m.y = x[m.order:]

	xMat := lm.AsMatrix(X)
	yMat := lm.AsMatrix(m.y)

	m.Coeff = lm.LinearRegression(xMat, yMat)
}

func lag(x []float64, shift, n int) []float64 {
	head := len(x) - n - shift
	return x[head : len(x)-shift]
}

func (m *AutoRegressiveModel) Predict(n int) []float64 {

	b := mat.NewDense(len(m.Coeff), 1, m.Coeff)

	for i := 0; i < n; i++ {
		terms := m.y[len(m.y)-m.order:]

		t := []float64{1.0}

		xMat := mat.NewDense(1, len(terms)+1, append(t, terms...))
		yHat := mat.Dense{}
		yHat.Mul(xMat, b)

		m.y = append(m.y, yHat.At(0, 0))
	}

	return m.y[len(m.y)-n:]
}
