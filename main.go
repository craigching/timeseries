package main

import (
	"fmt"

	"github.com/craigching/linear-regression/ar"
	"github.com/craigching/linear-regression/datasets"
	"github.com/craigching/linear-regression/lm"
)

func main() {

	X, y, _ := datasets.Mtcars()
	lm := lm.New()
	lm.Fit(X, y)
	fmt.Println("coeffs:", lm.Coeff)

	x := datasets.Lh()
	ar := ar.New()
	ar.Fit(x)
	pred := ar.Predict(10)
	fmt.Println("coeffs:", ar.Coeff)
	fmt.Println("pred:", pred)
}
