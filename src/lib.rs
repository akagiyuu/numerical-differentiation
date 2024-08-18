use cached::proc_macro::cached;
use nalgebra::{DMatrix, DVector, RowDVector};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

#[cached]
fn factorial(n: u128) -> u128 {
    match n {
        0 => 1,
        n => n * factorial(n - 1),
    }
}

/// https://en.wikipedia.org/wiki/Finite_difference_coefficient#Arbitrary_stencil_points
#[cached]
fn get_finite_difference_coefficients(stencils: Vec<i128>, d: u128) -> Vec<f64> {
    let n = stencils.len();

    let stencils = RowDVector::from_iterator(n, stencils.iter().map(|&i| i as f64));

    let mut stencil_matrix = DMatrix::from_element(n, n, 0f64);
    let mut pow = RowDVector::from_element(n, 1.);

    stencil_matrix.set_row(0, &pow);
    for mut row in stencil_matrix.row_iter_mut().skip(1) {
        pow = pow.component_mul(&stencils);
        row.copy_from(&pow);
    }

    stencil_matrix.try_inverse_mut();

    let mut factorial_column = DVector::from_element(n, 0.);
    factorial_column[d as usize] = factorial(d) as f64;

    let coefficients = stencil_matrix * factorial_column;

    coefficients.data.as_vec().to_owned()
}

pub fn get_epsilon(x: f64, d: u128) -> f64 {
    x * f64::EPSILON.powf(1. / (d as f64 + 1.))
}

pub fn differentiate_with_stencils<F: Fn(f64) -> f64 + Sync>(
    x: f64,
    f: F,
    d: u128,
    stencils: Vec<i128>,
) -> f64 {
    assert!((d as usize) < stencils.len());

    let epsilon = get_epsilon(x, d);
    let coefficients = get_finite_difference_coefficients(stencils.clone(), d);
    let numerator: f64 = stencils
        .into_par_iter()
        .zip(coefficients.into_par_iter())
        .map(|(stencil, coefficient)| coefficient * f(x + stencil as f64 * epsilon))
        .sum();
    let denominator = epsilon.powi(d as i32);

    numerator / denominator
}

#[cached]
fn get_stencils(d: u128) -> Vec<i128> {
    let start = -(d as i128) / 2;
    let end = (d as i128 + 1) / 2;

    (start..=end).collect()
}

pub fn differentiate<F: Fn(f64) -> f64 + Sync>(x: f64, f: F, d: u128) -> f64 {
    let stencils = get_stencils(d);

    differentiate_with_stencils(x, f, d, stencils)
}
