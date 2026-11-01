//! Hand-made panels for the integration tests.
#![allow(dead_code)]

use clm_data::{Bars, Panel};

/// Days of the hand panel.
pub const N: usize = 300;
/// The day with the volume spike (even).
pub const SPIKE: usize = 260;
/// Adjustment factor `adj_close / close` of every stock.
pub const ADJ: f64 = 0.5;
/// Gap factor: `O[k] = C[k-1] * GAP`.
pub const GAP: f64 = 1.003;

/// Stock `i` adjusted close: `100 * g_i^k * (1.01 if k odd)`, `ln g_i = 0.001*(i+1)`.
pub fn stock_close(i: usize, k: usize) -> f64 {
    let lg = 0.001 * (i + 1) as f64;
    100.0 * (lg * k as f64).exp() * if k % 2 == 1 { 1.01 } else { 1.0 }
}

/// Market close: `100 * exp(0.001 k) * (1.02 if k odd)`.
pub fn market_close(k: usize) -> f64 {
    100.0 * (0.001 * k as f64).exp() * if k % 2 == 1 { 1.02 } else { 1.0 }
}

/// Sector close: `100 * exp(0.0005 k)`.
pub fn sector_close(k: usize) -> f64 {
    100.0 * (0.0005 * k as f64).exp()
}

/// Stock `i` volume: 1000, except `1000*(i+1)` on `SPIKE`.
pub fn stock_volume(i: usize, k: usize) -> f64 {
    if k == SPIKE {
        1000.0 * (i + 1) as f64
    } else {
        1000.0
    }
}

/// Adjusted O/H/L/C of a close path: `O[k] = C[k-1]*GAP` (`O[0] = C[0]`),
/// `H = max(O, C) * 1.02`, `L = min(O, C) / 1.01`.
pub fn ohlc(c: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let o: Vec<f64> = (0..c.len())
        .map(|k| if k == 0 { c[0] } else { c[k - 1] * GAP })
        .collect();
    let h = (0..c.len()).map(|k| o[k].max(c[k]) * 1.02).collect();
    let l = (0..c.len()).map(|k| o[k].min(c[k]) / 1.01).collect();
    (o, h, l)
}

/// Raw bars whose adjusted prices are `c` (and OHLC from [`ohlc`]) with factor `adj`.
pub fn bars_from_adjusted(symbol: &str, dates: &[i32], c: &[f64], v: &[f64], adj: f64) -> Bars {
    let (o, h, l) = ohlc(c);
    Bars {
        symbol: symbol.into(),
        date: dates.to_vec(),
        open: o.iter().map(|x| x / adj).collect(),
        high: h.iter().map(|x| x / adj).collect(),
        low: l.iter().map(|x| x / adj).collect(),
        close: c.iter().map(|x| x / adj).collect(),
        adj_close: c.to_vec(),
        volume: v.to_vec(),
    }
}

pub fn dates(n: usize) -> Vec<i32> {
    clm_data::synthetic::weekday_calendar(n)
}

/// The raw bars of the hand panel with `n` days: market, one sector, `ns` stocks.
pub fn hand_bars(n: usize, ns: usize) -> (Bars, Vec<Bars>, Vec<Bars>) {
    let d = dates(n);
    let mc: Vec<f64> = (0..n).map(market_close).collect();
    let ec: Vec<f64> = (0..n).map(sector_close).collect();
    let ones = vec![1e6; n];
    let market = bars_from_adjusted("MKT", &d, &mc, &ones, 1.0);
    let sector = bars_from_adjusted("SEC", &d, &ec, &ones, 1.0);
    let stocks = (0..ns)
        .map(|i| {
            let c: Vec<f64> = (0..n).map(|k| stock_close(i, k)).collect();
            let v: Vec<f64> = (0..n).map(|k| stock_volume(i, k)).collect();
            bars_from_adjusted(&format!("S{i}"), &d, &c, &v, ADJ)
        })
        .collect();
    (market, vec![sector], stocks)
}

/// The hand panel: `N` days, 4 stocks in one sector.
pub fn hand_panel() -> Panel {
    panel_from(hand_bars(N, 4))
}

pub fn panel_from((m, sec, st): (Bars, Vec<Bars>, Vec<Bars>)) -> Panel {
    let ss = vec![0; st.len()];
    Panel::from_bars(&m, &sec, &st, &ss).unwrap().0
}

/// Removes the row of `date_index` from bars.
pub fn drop_row(b: &mut Bars, k: usize) {
    b.date.remove(k);
    b.open.remove(k);
    b.high.remove(k);
    b.low.remove(k);
    b.close.remove(k);
    b.adj_close.remove(k);
    b.volume.remove(k);
}

/// `(a - b).abs() <= tol * max(1, |b|)`.
pub fn close_to(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol * b.abs().max(1.0)
}

#[track_caller]
pub fn assert_close(a: f64, b: f64, tol: f64, what: &str) {
    assert!(close_to(a, b, tol), "{what}: got {a}, expected {b}");
}

/// Real bars directory, if downloaded.
pub fn real_dir() -> Option<std::path::PathBuf> {
    let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/bars");
    p.join("universe.json").exists().then_some(p)
}
