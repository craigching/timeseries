package ar

import (
	"github.com/craigching/timeseries/lm"
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
	c := lm.LinearRegression(X, m.y)
	m.Coeff = c
}

func lag(x []float64, shift, n int) []float64 {
	head := len(x) - n - shift
	return x[head : len(x)-shift]
}

func (m *AutoRegressiveModel) Predict(n int) []float64 {

	// Reverse the coefficients when multiplying to match what R's ar() is doing
	coeff := make([]float64, len(m.Coeff)-1)
	copy(coeff, m.Coeff[1:])
	for i, j := 0, len(coeff)-1; i < j; i, j = i+1, j-1 {
		coeff[i], coeff[j] = coeff[j], coeff[i]
	}
	xint := m.Coeff[0]

	for i := 0; i < n; i++ {
		terms := m.y[len(m.y)-m.order:]
		sum := xint
		for i := 0; i < m.order; i++ {
			sum += terms[i] * coeff[i]
		}
		m.y = append(m.y, sum)
	}

	return m.y[len(m.y)-n:]
}
