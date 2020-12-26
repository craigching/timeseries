package ar

import "github.com/craigching/linear-regression/lm"

type AutoRegressiveModel struct {
	lm.LinearModel
}

func New(x []float64) lm.Model {
	return &AutoRegressiveModel{
		LinearModel: lm.LinearModel{
			X: [][]float64{x[:len(x)-1]},
			Y: x[1:],
		},
	}
}

func (m *AutoRegressiveModel) Fit() {
	c := lm.LinearRegression(m.X, m.Y)
	m.C = c
}

func (m *AutoRegressiveModel) Predict(n int) []float64 {
	pred := []float64{}

	for i := 0; i < n; i++ {
		var p float64
		if len(pred) == 0 {
			p = m.Y[len(m.Y)-1]
		} else {
			p = pred[i-1]
		}
		// TODO need to handle order > 1
		pred = append(pred, p*m.C[1]+m.C[0])
	}

	return pred
}
