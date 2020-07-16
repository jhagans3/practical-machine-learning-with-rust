use rand::seq::SliceRandom;
use rand::thread_rng;
use std::error::Error;
use std::fs::File;
use std::io::prelude::*;
use std::io::BufReader;
use std::path::Path;
use std::vec::Vec;

pub struct BostonHousing {
    // CRIM - per capita crime rate by town
    crim: f64,
    // ZN - proportion of residential land zoned for lots over 25,000 sq. ft.
    zn: f64,
    // INDUS - proportion of non-retail business acres per town
    indus: f64,
    // CHAS - Charles River dummy variable (1 if tract bounds river; 0 otherwise)
    chas: f64,
    // NOX - nitric oxides concentration (parts per 10 million)
    nox: f64,
    // RM - average number of rooms per dwelling
    rm: f64,
    // AGE - proportion of owner-occupied units built prior to 1940
    age: f64,
    // DIS - weighted distances to five Boston employment centers
    dis: f64,
    // RAD - index of accessibility to radial highways
    rad: f64,
    // TAX - full-value property-tax rate per $10,000
    tax: f64,
    // PTRATIO - pupil-teacher ratio by town
    ptratio: f64,
    // Black - 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
    black: f64,
    // LSTAT - % lower status of the population
    lstat: f64,
    // MEDV - Median value of owner-occupied homes in $1000’s,
    medv: f64,
}

impl BostonHousing {
    pub fn new(v: Vec<&str>) -> BostonHousing {
        let f64_formatted: Vec<f64> = v.iter().map(|s| s.parse().unwrap()).collect();
        BostonHousing {
            crim: f64_formatted[0],
            zn: f64_formatted[1],
            indus: f64_formatted[2],
            chas: f64_formatted[3],
            nox: f64_formatted[4],
            rm: f64_formatted[5],
            age: f64_formatted[6],
            dis: f64_formatted[7],
            rad: f64_formatted[8],
            tax: f64_formatted[9],
            ptratio: f64_formatted[10],
            black: f64_formatted[11],
            lstat: f64_formatted[12],
            medv: f64_formatted[13],
        }
    }

    pub fn into_feature_vector(&self) -> Vec<f64> {
        vec![
            self.crim,
            self.zn,
            self.indus,
            self.chas,
            self.nox,
            self.rm,
            self.age,
            self.dis,
            self.rad,
            self.tax,
            self.ptratio,
            self.black,
            self.lstat,
        ]
    }

    pub fn into_targets(&self) -> f64 {
        self.medv
    }
}

fn get_boston_record(s: String) -> BostonHousing {
    let v: Vec<&str> = s.split_whitespace().collect();
    let b: BostonHousing = BostonHousing::new(v);
    b
}

pub fn get_boston_records_from_file(filename: impl AsRef<Path>) -> Vec<BostonHousing> {
    let file = File::open(filename).expect("no such file");
    let buf = BufReader::new(file);
    buf.lines()
        .enumerate()
        .map(|(n, l)| l.expect(&format!("Could not parse line no {}", n)))
        .map(|r| get_boston_record(r))
        .collect()
}

fn main() {
    println!("Hello, world!");
}
