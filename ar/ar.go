package ar

import "github.com/craigching/timeseries/lm"

type AutoRegressiveModel struct {
	lm.LinearModel
	y []float64
}

func New() *AutoRegressiveModel {
	return &AutoRegressiveModel{
		LinearModel: lm.LinearModel{},
	}
}

func (m *AutoRegressiveModel) Fit(x []float64) {
	X := [][]float64{x[:len(x)-1]}
	// Save y so we can predict from it
	m.y = x[1:]
	c := lm.LinearRegression(X, m.y)
	m.Coeff = c
}

func (m *AutoRegressiveModel) Predict(n int) []float64 {
	pred := []float64{}

	for i := 0; i < n; i++ {
		var p float64
		if len(pred) == 0 {
			p = m.y[len(m.y)-1]
		} else {
			p = pred[i-1]
		}
		// TODO need to handle order > 1
		pred = append(pred, p*m.Coeff[1]+m.Coeff[0])
	}

	return pred
}
