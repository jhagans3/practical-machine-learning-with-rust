use rand::seq::SliceRandom;
use rand::thread_rng;
use rusty_machine;
use rusty_machine::analysis::score::neg_mean_squared_error;
use rusty_machine::learning::lin_reg::LinRegressor;
use rusty_machine::learning::SupModel;
use rusty_machine::linalg::Matrix;
use rusty_machine::linalg::Vector;
use std::error::Error;
use std::fs::File;
use std::io::prelude::*;
use std::io::BufReader;
use std::path::Path;
use std::vec::Vec;

// for regression
pub fn r_squared_score(y_test: &[f64], y_preds: &[f64]) -> f64 {
    let model_variance: f64 = y_test
        .iter()
        .zip(y_preds.iter())
        .fold(0., |v, (y_i, y_i_hat)| v + (y_i - y_i_hat).powi(2));

    // get the mean for the actual values to be used later
    let y_test_mean = y_test.iter().sum::<f64>() as f64 / y_test.len() as f64;

    // finding the variance
    let variance = y_test
        .iter()
        .fold(0., |v, &x| v + (x - y_test_mean).powi(2));
    let r2_calculated: f64 = 1.0 - (model_variance / variance);
    r2_calculated
}

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

pub fn run() -> Result<(), Box<dyn Error>> {
    // Get all the data
    let filename = "data/housing.csv";
    let mut data = get_boston_records_from_file(&filename); // file must be in the folder data

    // shuffle the data.
    data.shuffle(&mut thread_rng());

    // separate out to train and test datasets.
    let test_size: f64 = 0.2;
    let test_size: f64 = data.len() as f64 * test_size;
    let test_size = test_size.round() as usize;
    let (test_data, train_data) = data.split_at(test_size);
    let train_size = train_data.len();
    let test_size = test_data.len();

    // differentiate the features and the targets.
    let boston_x_train: Vec<f64> = train_data
        .iter()
        .flat_map(|r| r.into_feature_vector())
        .collect();
    let boston_y_train: Vec<f64> = train_data.iter().map(|r| r.into_targets()).collect();

    let boston_x_test: Vec<f64> = test_data
        .iter()
        .flat_map(|r| r.into_feature_vector())
        .collect();
    let boston_y_test: Vec<f64> = test_data.iter().map(|r| r.into_targets()).collect();

    // Convert the data into matrices for rusty machine
    let boston_x_train = Matrix::new(train_size, 13, boston_x_train);
    let boston_y_train = Vector::new(boston_y_train);
    let boston_x_test = Matrix::new(test_size, 13, boston_x_test);
    // let boston_y_test = Vector::new(boston_y_test);
    let boston_y_test = Matrix::new(test_size, 1, boston_y_test);

    // Create a linear regression model
    let mut lin_model = LinRegressor::default();
    println!("{:?}", lin_model);

    // Train the model
    lin_model.train(&boston_x_train, &boston_y_train);

    // Now we will predict
    let predictions = lin_model.predict(&boston_x_test).unwrap();
    let predictions = Matrix::new(test_size, 1, predictions);
    let acc = neg_mean_squared_error(&predictions, &boston_y_test);
    println!("linear regression error: {:?}", acc);
    println!(
        "linear regression R2 score: {:?}",
        r_squared_score(&boston_y_test.data(), &predictions.data())
    );

    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    run()?;

    Ok(())
}
