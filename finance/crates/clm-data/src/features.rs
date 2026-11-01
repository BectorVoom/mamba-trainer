//! Feature set v1 (CONTRACTS C2): 20 point-in-time features per (stock, day).
//!
//! Every value of row `(s, t)` reads bars of days `t - K ..= t` (`K = 252`) of the
//! stock, the market index and the stock's sector ETF, plus same-day features of the
//! other stocks (cross-sectional z-scores). Computation is in `f64`, storage `f32`.
//! Every value is computed directly from its own window (no running sums), so a value
//! depends only on the bars it reads, bit for bit — the truncation audit relies on it.
//!
//! Validity of row `(s, t)`: every bar any of the 20 features reads exists and every
//! value is finite. Concretely the stock needs bars `t-251 ..= t` (hi52 is the longest
//! window), positive volume on `t-19 ..= t` (relvol, amihud), the market bars
//! `t-20 ..= t` and the sector bars `t-20`, `t-1`, `t`. The first possible valid day is
//! `t = 251`. Invalid rows hold NaN in every column.
//!
//! Cross-sectional features (18, 19): the cross-section of day `t` is the set of stocks
//! whose features 0..=17 are all valid and finite on `t`; for those stocks
//! `z = (x - mean) / sd` (population sd, securities in index order), and `z = 0` when the
//! cross-section has fewer than 3 stocks or `sd == 0`.
//!
//! **amihud** uses the traded value `close * volume` with the *reported* close, not the
//! adjusted close of C1: `adj_close` is scaled by every later dividend, so
//! `adj_close * volume` would carry the stock's future dividend yield into a level-type
//! feature (C1 forbids levels for exactly this reason). Ratio features are unaffected.

use crate::bars::{Panel, Series};

/// Number of features in set v1.
pub const NUM_FEATURES: usize = 20;
/// Maximum lookback `K` in days.
pub const LOOKBACK: usize = 252;
/// Annualisation factor for realised volatility.
pub const TRADING_DAYS: f64 = 252.0;

/// Feature names in column order.
pub const FEATURE_NAMES: [&str; NUM_FEATURES] = [
    "r1",
    "r5",
    "r20",
    "r60",
    "gap",
    "intraday",
    "range",
    "rv20",
    "rv60",
    "relvol",
    "mom",
    "hi52",
    "amihud",
    "mkt_r1",
    "mkt_r20",
    "mkt_rv20",
    "sec_r1",
    "sec_r20",
    "xs_r20",
    "xs_relvol",
];

/// Column indices.
pub mod col {
    #![allow(missing_docs)]
    pub const R1: usize = 0;
    pub const R5: usize = 1;
    pub const R20: usize = 2;
    pub const R60: usize = 3;
    pub const GAP: usize = 4;
    pub const INTRADAY: usize = 5;
    pub const RANGE: usize = 6;
    pub const RV20: usize = 7;
    pub const RV60: usize = 8;
    pub const RELVOL: usize = 9;
    pub const MOM: usize = 10;
    pub const HI52: usize = 11;
    pub const AMIHUD: usize = 12;
    pub const MKT_R1: usize = 13;
    pub const MKT_R20: usize = 14;
    pub const MKT_RV20: usize = 15;
    pub const SEC_R1: usize = 16;
    pub const SEC_R20: usize = 17;
    pub const XS_R20: usize = 18;
    pub const XS_RELVOL: usize = 19;
}

/// Feature rows of every (stock, day): `values[(s * num_days + t) * num_features + f]`.
/// Equality is bitwise on the values (NaN rows compare equal).
#[derive(Debug, Clone)]
pub struct FeatureTensor {
    /// Stocks.
    pub num_securities: usize,
    /// Calendar days.
    pub num_days: usize,
    /// Columns.
    pub num_features: usize,
    /// `[s][t][f]` row-major; NaN in invalid rows.
    pub values: Vec<f32>,
    /// `[s][t]`: the row is valid (C2).
    pub valid: Vec<bool>,
    /// Column names.
    pub names: Vec<String>,
}

impl PartialEq for FeatureTensor {
    fn eq(&self, o: &Self) -> bool {
        (self.num_securities, self.num_days, self.num_features)
            == (o.num_securities, o.num_days, o.num_features)
            && self.valid == o.valid
            && self.names == o.names
            && self.values.len() == o.values.len()
            && self
                .values
                .iter()
                .zip(&o.values)
                .all(|(a, b)| a.to_bits() == b.to_bits())
    }
}

impl FeatureTensor {
    /// All rows invalid (NaN).
    pub fn new_invalid(num_securities: usize, num_days: usize, names: Vec<String>) -> Self {
        let num_features = names.len();
        Self {
            num_securities,
            num_days,
            num_features,
            values: vec![f32::NAN; num_securities * num_days * num_features],
            valid: vec![false; num_securities * num_days],
            names,
        }
    }

    /// Row offset of `(s, t)` in `valid`.
    #[inline]
    pub fn row_index(&self, s: usize, t: usize) -> usize {
        debug_assert!(s < self.num_securities && t < self.num_days);
        s * self.num_days + t
    }

    /// The feature row `(s, t)`.
    #[inline]
    pub fn row(&self, s: usize, t: usize) -> &[f32] {
        let i = self.row_index(s, t) * self.num_features;
        &self.values[i..i + self.num_features]
    }

    /// The feature row `(s, t)`, mutable.
    #[inline]
    pub fn row_mut(&mut self, s: usize, t: usize) -> &mut [f32] {
        let i = self.row_index(s, t) * self.num_features;
        &mut self.values[i..i + self.num_features]
    }

    /// One value.
    #[inline]
    pub fn get(&self, s: usize, t: usize, f: usize) -> f32 {
        self.row(s, t)[f]
    }

    /// Whether row `(s, t)` is valid.
    #[inline]
    pub fn is_valid(&self, s: usize, t: usize) -> bool {
        self.valid[self.row_index(s, t)]
    }

    /// Number of valid rows.
    pub fn num_valid(&self) -> usize {
        self.valid.iter().filter(|&&v| v).count()
    }

    /// Marks row `(s, t)` invalid and fills it with NaN.
    pub fn invalidate(&mut self, s: usize, t: usize) {
        let i = self.row_index(s, t);
        self.valid[i] = false;
        self.row_mut(s, t).fill(f32::NAN);
    }
}

/// One series plus derived columns the feature kernels share: `r1[k] = log(C[k]/C[k-1])`
/// (NaN unless both bars exist) and prefix counts of present days. `r1[k]` is the same
/// expression wherever it is evaluated, so caching changes no bit of any feature.
#[derive(Debug, Clone)]
pub struct SeriesCache<'a> {
    /// The series.
    pub x: &'a Series,
    r1: Vec<f64>,
    count: Vec<u32>,
}

impl<'a> SeriesCache<'a> {
    /// Builds the derived columns.
    pub fn new(x: &'a Series) -> Self {
        let n = x.len();
        let r1 = (0..n)
            .map(|k| {
                if k >= 1 && x.present[k] && x.present[k - 1] {
                    (x.close[k] / x.close[k - 1]).ln()
                } else {
                    f64::NAN
                }
            })
            .collect();
        let mut count = Vec::with_capacity(n + 1);
        count.push(0u32);
        for k in 0..n {
            count.push(count[k] + x.present[k] as u32);
        }
        Self { x, r1, count }
    }

    /// True if the bar of day `t` exists.
    #[inline]
    pub fn has(&self, t: usize) -> bool {
        self.x.has(t)
    }

    /// True if every day in `lo ..= hi` has a bar (`lo <= hi`).
    #[inline]
    pub fn has_all(&self, lo: usize, hi: usize) -> bool {
        hi < self.x.len() && (self.count[hi + 1] - self.count[lo]) as usize == hi + 1 - lo
    }
}

/// `log(C[t] / C[t-n])`, if both bars exist.
#[inline]
pub fn log_return(c: &SeriesCache, t: usize, n: usize) -> Option<f64> {
    let x = c.x;
    (t >= n && x.has(t) && x.has(t - n)).then(|| (x.close[t] / x.close[t - n]).ln())
}

/// Population std of `r1[k]`, `k = t-n+1 ..= t`, times `sqrt(252)`; reads `C[t-n ..= t]`.
pub fn realized_vol(c: &SeriesCache, t: usize, n: usize) -> Option<f64> {
    if t < n || !c.has_all(t - n, t) {
        return None;
    }
    let r = &c.r1[t - n + 1..=t];
    let mean = r.iter().sum::<f64>() / n as f64;
    let ss: f64 = r.iter().map(|v| (v - mean) * (v - mean)).sum();
    Some((ss / n as f64).sqrt() * TRADING_DAYS.sqrt())
}

/// `log(V[t] / mean(V[t-19 ..= t]))`; requires positive volume on every day of the window.
pub fn relvol(c: &SeriesCache, t: usize) -> Option<f64> {
    if t < 19 || !c.has_all(t - 19, t) {
        return None;
    }
    let v = &c.x.volume[t - 19..=t];
    if v.iter().any(|&x| x <= 0.0) {
        return None;
    }
    Some((c.x.volume[t] / (v.iter().sum::<f64>() / 20.0)).ln())
}

/// `log(1e-12 + mean_{k=t-19..t} |r1[k]| / DV[k])` with `DV = close * volume` (reported
/// close, see the module docs); reads bars `t-20 ..= t`, requires positive volume.
pub fn amihud(c: &SeriesCache, t: usize) -> Option<f64> {
    if t < 20 || !c.has_all(t - 20, t) {
        return None;
    }
    let x = c.x;
    let mut sum = 0.0;
    for k in t - 19..=t {
        if x.volume[k] <= 0.0 {
            return None;
        }
        sum += c.r1[k].abs() / x.dollar_volume[k];
    }
    Some((1e-12 + sum / 20.0).ln())
}

/// `log(C[t] / max(H[t-251 ..= t]))`.
pub fn hi52(c: &SeriesCache, t: usize) -> Option<f64> {
    if t < LOOKBACK - 1 || !c.has_all(t + 1 - LOOKBACK, t) {
        return None;
    }
    let m = c.x.high[t + 1 - LOOKBACK..=t]
        .iter()
        .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    Some((c.x.close[t] / m).ln())
}

/// Caches of the market, every sector ETF and every stock of a panel.
pub struct PanelCache<'a> {
    /// The panel.
    pub panel: &'a Panel,
    /// Market.
    pub market: SeriesCache<'a>,
    /// Sector ETFs.
    pub sectors: Vec<SeriesCache<'a>>,
    /// Stocks.
    pub stocks: Vec<SeriesCache<'a>>,
}

impl<'a> PanelCache<'a> {
    /// Builds every cache.
    pub fn new(panel: &'a Panel) -> Self {
        Self {
            panel,
            market: SeriesCache::new(&panel.market),
            sectors: panel.sectors.iter().map(SeriesCache::new).collect(),
            stocks: panel.stocks.iter().map(SeriesCache::new).collect(),
        }
    }
}

/// Features 0..=17 of stock `s` on day `t` (`None` = invalid), in `f64`.
pub fn base_row(pc: &PanelCache, s: usize, t: usize) -> Option<[f64; 18]> {
    let c = &pc.stocks[s];
    let m = &pc.market;
    let sec = &pc.sectors[pc.panel.sector_of[s]];
    let x = c.x;
    if t < 1 || !x.has(t) || !x.has(t - 1) {
        return None;
    }
    let r1 = log_return(c, t, 1)?;
    let r20 = log_return(c, t, 20)?;
    let mom = (t >= 120 && x.has(t - 20) && x.has(t - 120))
        .then(|| (x.close[t - 20] / x.close[t - 120]).ln())?;
    let row = [
        r1,
        log_return(c, t, 5)?,
        r20,
        log_return(c, t, 60)?,
        (x.open[t] / x.close[t - 1]).ln(),
        (x.close[t] / x.open[t]).ln(),
        (x.high[t] / x.low[t]).ln(),
        realized_vol(c, t, 20)?,
        realized_vol(c, t, 60)?,
        relvol(c, t)?,
        mom,
        hi52(c, t)?,
        amihud(c, t)?,
        log_return(m, t, 1)?,
        log_return(m, t, 20)?,
        realized_vol(m, t, 20)?,
        r1 - log_return(sec, t, 1)?,
        r20 - log_return(sec, t, 20)?,
    ];
    row.iter().all(|v| v.is_finite()).then_some(row)
}

/// Population z-scores of `xs` (0 if fewer than 3 values or `sd == 0`).
pub fn zscores(xs: &[f64]) -> Vec<f64> {
    let n = xs.len();
    if n < 3 {
        return vec![0.0; n];
    }
    let mean = xs.iter().sum::<f64>() / n as f64;
    let var = xs.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n as f64;
    let sd = var.sqrt();
    if sd == 0.0 || !sd.is_finite() {
        return vec![0.0; n];
    }
    xs.iter().map(|x| (x - mean) / sd).collect()
}

/// Computes feature set v1 for every stock and day of the panel.
pub fn compute_features(panel: &Panel) -> FeatureTensor {
    let ns = panel.num_securities();
    let nd = panel.num_days();
    let names = FEATURE_NAMES.iter().map(|s| s.to_string()).collect();
    let mut out = FeatureTensor::new_invalid(ns, nd, names);
    let pc = PanelCache::new(panel);
    let mut day: Vec<(usize, [f64; 18])> = Vec::with_capacity(ns);
    for t in 0..nd {
        day.clear();
        day.extend((0..ns).filter_map(|s| base_row(&pc, s, t).map(|r| (s, r))));
        let z_r20 = zscores(&day.iter().map(|(_, r)| r[col::R20]).collect::<Vec<_>>());
        let z_rv = zscores(&day.iter().map(|(_, r)| r[col::RELVOL]).collect::<Vec<_>>());
        for (i, (s, base)) in day.iter().enumerate() {
            let full: Vec<f64> = base.iter().copied().chain([z_r20[i], z_rv[i]]).collect();
            if !full.iter().all(|&v| (v as f32).is_finite()) {
                continue;
            }
            let ri = out.row_index(*s, t);
            out.valid[ri] = true;
            for (dst, v) in out.row_mut(*s, t).iter_mut().zip(&full) {
                *dst = *v as f32;
            }
        }
    }
    out
}
