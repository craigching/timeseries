package ar

import (
	"testing"

	"github.com/craigching/timeseries/datasets"
	"github.com/stretchr/testify/assert"
)

func TestHappyPath(t *testing.T) {

	expectedCoeff := []float64{0.9998651719436421, 0.5859869716709594}
	expectedPred := []float64{2.6992273897894243, 2.5815772559376553, 2.512635810285174, 2.4722370213246583, 2.4485638573225117, 2.4346916916390238, 2.426562783279639, 2.421799348887132, 2.4190080383927137, 2.4173723668090963}

	x := datasets.Lh()
	ar := New(1)
	ar.Fit(x)
	pred := ar.Predict(10)

	assert.Equal(t, expectedCoeff, ar.Coeff)
	assert.Equal(t, expectedPred, pred)
}
