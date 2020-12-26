package main

import (
	"fmt"

	"github.com/craigching/timeseries/ar"
	"github.com/craigching/timeseries/datasets"
	"github.com/craigching/timeseries/lm"
)

func main() {

	X, y, _ := datasets.Mtcars()
	lm := lm.New()
	lm.Fit(X, y)
	fmt.Println("coeffs:", lm.Coeff)

	x := datasets.Lh()
	ar := ar.New(1)
	ar.Fit(x)
	pred := ar.Predict(10)
	fmt.Println("coeffs:", ar.Coeff)
	fmt.Println("pred:", pred)
}
