use rand::seq::SliceRandom;
use rand::thread_rng;
use std::error::Error;
use std::fs::File;
use std::io::prelude::*;
use std::io::BufReader;
use std::path::Path;
use std::vec::Vec;
use tensorflow as tf;
use tf::{DataType, Graph, Tensor};
use tf::{Session, SessionOptions, SessionRunArgs};

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

    // Define graph.
    let dim = (boston_y_train.len() as u64, 13);
    let test_dim = (boston_y_test.len() as u64, dim.1);
    let X_train = <Tensor<f64>>::new(&[dim.0, dim.1]).with_values(&boston_x_train)?;
    let y_train = <Tensor<f64>>::new(&[dim.0, 1]).with_values(&boston_y_train)?;
    let X_test = <Tensor<f64>>::new(&[test_dim.0, test_dim.1]).with_values(&boston_x_test)?;
    // let y_test = <Tensor<f64>>::new(&[test_dim.0, 1]).with_values(&boston_y_test)?;

    let mut output_array = vec![0.0; (dim.0 * dim.1) as usize];
    transpose::transpose(
        &boston_x_train,
        &mut output_array,
        dim.1 as usize,
        dim.0 as usize,
    );
    let XT = <Tensor<f64>>::new(&[dim.1, dim.0]).with_values(&output_array[..])?;

    let mut graph = Graph::new();

    // A tensorflow graph represents the data flow of the computations.
    // We can code the specific computations that will go as part of the graph
    let XT_const = {
        let mut op = graph.new_operation("Const", "XT")?;
        op.set_attr_tensor("value", XT)?;
        op.set_attr_type("dtype", DataType::Double)?; // check the enums https://github.com/tensorflow/rust/blob/ddff61850be1c8044ac86350caeed5a55824ebe4/src/lib.rs#L297
        op.finish()?
    };
    let X_const = {
        let mut op = graph.new_operation("Const", "X_train")?;
        op.set_attr_tensor("value", X_train)?;
        op.set_attr_type("dtype", DataType::Double)?; // check the enums https://github.com/tensorflow/rust/blob/ddff61850be1c8044ac86350caeed5a55824ebe4/src/lib.rs#L297
        op.finish()?
    };
    // operation types https://github.com/malmaud/TensorFlow.jl/blob/063511525902bdf84a461035758ef9a73ba4a635/src/ops/op_names.txt
    let y_const = {
        let mut op = graph.new_operation("Const", "y_train")?;
        op.set_attr_tensor("value", y_train)?;
        op.set_attr_type("dtype", DataType::Double)?;
        op.finish()?
    };
    let mul = {
        let mut op = graph.new_operation("MatMul", "mul")?;
        op.add_input(XT_const.clone());
        op.add_input(X_const.clone());
        op.finish()?
    };
    let inverse = {
        let mut op = graph.new_operation("MatrixInverse", "mul_inv")?;
        op.add_input(mul);
        op.finish()?
    };
    let mul2 = {
        let mut op = graph.new_operation("MatMul", "mul2")?;
        op.add_input(inverse);
        op.add_input(XT_const.clone());
        op.finish()?
    };
    let theta = {
        let mut op = graph.new_operation("MatMul", "theta")?;
        op.add_input(mul2);
        op.add_input(y_const);
        op.finish()?
    };

    // running predictions
    // y = X_test .* theta
    let X_test_const = {
        let mut op = graph.new_operation("Const", "X_test")?;
        op.set_attr_tensor("value", X_test)?;
        op.set_attr_type("dtype", DataType::Double)?;
        op.finish()?
    };
    let predictions = {
        let mut op = graph.new_operation("MatMul", "preds")?;
        op.add_input(X_test_const);
        op.add_input(theta);
        op.finish()?
    };

    // Run graph.
    let session = Session::new(&SessionOptions::new(), &graph)?;
    let mut args = SessionRunArgs::new();
    let preds_token = args.request_fetch(&predictions, 0);
    session.run(&mut args)?;
    let preds_token_res: Tensor<f64> = args.fetch::<f64>(preds_token)?;
    // println!("Now the preds", );
    // println!("{:?}", &preds_token_res[..]);
    println!(
        "r-squared error score: {:?}",
        r_squared_score(&preds_token_res.to_vec(), &boston_y_test)
    );

    Ok(())
}

// ../practical-machine-learning-with-rust/supervised-learning/data/housing.csv
// ../practical-machine-learning-with-rust/supervised-learning$ cargo run --bin tf-regression
fn main() -> Result<(), Box<dyn Error>> {
    run()?;

    Ok(())
}
