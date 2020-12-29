package ar

import "fmt"

type ArmaModel struct {
	p int
	q int
}

func NewARMA(p, q int) *ArmaModel {
	return &ArmaModel{
		p: p,
		q: q,
	}
}

func (m *ArmaModel) Fit(x []float64) {
	ar := New(4)
	ar.Fit(x)
	fmt.Printf("Coeff: %v\n", ar.Coeff)
	fmt.Println(ar.Resid)
	fmt.Printf("len of residuals: %d\n", len(ar.Resid))
	fmt.Printf("len of data points: %d\n", len(x))

	// var X [][]float64
	// // n is the length of the lag arrays
	// n := len(x) - m.order
	// // build lags
	// for i := 0; i < m.order; i++ {
	// 	X = append(X, lag(x, i+1, n))
	// }
	// // Save y so we can predict from it
	// m.y = x[m.order:]
}
