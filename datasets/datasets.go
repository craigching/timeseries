package datasets

import (
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"strconv"
)

func Mtcars() ([][]float64, []float64, error) {
	f, err := os.Open("./mtcars.csv")
	if err != nil {
		fmt.Println("error reading csv", err)
		os.Exit(-1)
	}
	defer f.Close()

	csvr := csv.NewReader(f)

	i := 0

	mpg := []float64{}
	wt := []float64{}
	qsec := []float64{}

	for {
		row, err := csvr.Read()

		if err != nil {
			if err == io.EOF {
				break
			} else {
				return [][]float64{}, []float64{}, err
			}
		}

		i++
		if i == 1 {
			continue
		}

		// mpg == 1
		// wt == 6
		// qsec == 7

		x1, err := strconv.ParseFloat(row[1], 64)
		if err != nil {
			return [][]float64{}, []float64{}, fmt.Errorf("error parsing mpg: %s", err)
		}

		x2, err := strconv.ParseFloat(row[6], 64)
		if err != nil {
			return [][]float64{}, []float64{}, fmt.Errorf("error parsing wt: %s", err)
		}

		x3, err := strconv.ParseFloat(row[7], 64)
		if err != nil {
			return [][]float64{}, []float64{}, fmt.Errorf("error parsing qsec: %s", err)
		}

		mpg = append(mpg, x1)
		wt = append(wt, x2)
		qsec = append(qsec, x3)
	}

	return [][]float64{wt, qsec}, mpg, nil
}

func Lh() []float64 {
	return []float64{2.4, 2.4, 2.4, 2.2, 2.1, 1.5, 2.3, 2.3, 2.5, 2.0, 1.9, 1.7, 2.2, 1.8, 3.2, 3.2, 2.7, 2.2, 2.2, 1.9, 1.9, 1.8, 2.7, 3.0, 2.3, 2.0, 2.0, 2.9, 2.9, 2.7, 2.7, 2.3, 2.6, 2.4, 1.8, 1.7, 1.5, 1.4, 2.1, 3.3, 3.5, 3.5, 3.1, 2.6, 2.1, 3.4, 3.0, 2.9}
}
