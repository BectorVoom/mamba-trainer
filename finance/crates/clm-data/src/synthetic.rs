//! Deterministic synthetic panels for tests.
//!
//! Process (daily log returns, `N` = independent standard normals, everything seeded):
//!
//! * market `m[k] = 0.0003 + 0.01*N`;
//! * sector `j`: `e_j[k] = beta_j * m[k] + 0.006*N`, `beta_j` in `[0.8, 1.2]`;
//! * stock `i` (sector `j`): base `b_i[k] = beta_i * e_j[k] + 0.012*N`, `beta_i` in
//!   `[0.7, 1.3]`; ETFs and the market use their own return as base;
//! * every series splits its close-to-close return into a gap and an intraday part:
//!   `gap[k] = 0.25*b[k] + 0.002*N`, `intraday[k] = b[k] - gap[k] + drift*z[k-1]`
//!   (the planted term only for stocks), and
//!   `O[k] = C[k-1]*exp(gap[k])`, `C[k] = O[k]*exp(intraday[k])`,
//!   `H = max(O,C)*exp(|0.004*N|)`, `L = min(O,C)*exp(-|0.004*N|)`;
//! * volume `V[k] = V0 * exp(0.3*N + loading*z[k])`, `V0` per series in `[1e6, 1e7]`;
//! * `adj_close = 0.9 * close` for stocks (a constant adjustment factor: exercises the C1
//!   adjustment without changing any ratio), `= close` for ETFs.
//!
//! **Planted signal** ([`PlantedSignal`]): a hidden regime `z[s][t]` in `{-1, +1}`, i.i.d.
//! fair coin per stock and day. It raises log volume of day `t` by `loading * z[t]` —
//! visible at the close of `t` in `relvol` (feature 9) and `xs_relvol` (19) — and adds
//! `drift * z[t]` to the intraday log return of day `t+1`, i.e. to `log(C[t+1]/O[t+1])`,
//! which every label `r_h(t) = log(O[t+1+h]/O[t+1])` contains. So `E[r_h(t) | z[t]] =
//! drift * z[t]` above the unconditional drift. Without a signal, `z` is still drawn (and
//! returned) but has no effect. Dates are weekdays from 2000-01-03.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::bars::{Bars, Panel};

/// A standard normal draw (Box–Muller).
pub fn normal(rng: &mut StdRng) -> f64 {
    let u1: f64 = 1.0 - rng.random::<f64>(); // (0, 1]
    let u2: f64 = rng.random::<f64>();
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// The planted predictable signal (see the module docs).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlantedSignal {
    /// Added to the next day's intraday log return per unit of `z`.
    pub drift: f64,
    /// Added to log volume of the same day per unit of `z`.
    pub loading: f64,
}

impl Default for PlantedSignal {
    /// `drift = 0.004`, `loading = 0.5`.
    fn default() -> Self {
        Self {
            drift: 0.004,
            loading: 0.5,
        }
    }
}

/// Synthetic panel parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct SyntheticConfig {
    /// RNG seed.
    pub seed: u64,
    /// Calendar length.
    pub num_days: usize,
    /// Sector ETFs.
    pub num_sectors: usize,
    /// Stocks per sector.
    pub stocks_per_sector: usize,
    /// Probability that a stock's bar is absent on a day (ETFs and market are complete).
    pub missing_prob: f64,
    /// Planted signal, if any.
    pub signal: Option<PlantedSignal>,
}

impl Default for SyntheticConfig {
    /// 600 days, 2 sectors x 4 stocks, no missing bars, no signal, seed 7.
    fn default() -> Self {
        Self {
            seed: 7,
            num_days: 600,
            num_sectors: 2,
            stocks_per_sector: 4,
            missing_prob: 0.0,
            signal: None,
        }
    }
}

/// A synthetic panel and its hidden regime.
#[derive(Debug, Clone)]
pub struct SyntheticPanel {
    /// The panel (stocks `S000..`, sectors `SEC0..`, market `MKT`).
    pub panel: Panel,
    /// Hidden regime `z[s][t]` in `{-1, +1}`, `[s * num_days + t]`.
    pub regime: Vec<i8>,
    /// The raw bars (market, sectors, stocks), e.g. to write as Parquet.
    pub bars: Vec<Bars>,
}

/// Weekdays from 2000-01-03 (a Monday) as `yyyymmdd`.
pub fn weekday_calendar(n: usize) -> Vec<i32> {
    let (mut y, mut m, mut d, mut wd) = (2000i32, 1i32, 3i32, 0u32); // wd 0 = Monday
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        if wd < 5 {
            out.push(y * 10_000 + m * 100 + d);
        }
        wd = (wd + 1) % 7;
        let leap = (y % 4 == 0 && y % 100 != 0) || y % 400 == 0;
        let dim = match m {
            2 if leap => 29,
            2 => 28,
            4 | 6 | 9 | 11 => 30,
            _ => 31,
        };
        d += 1;
        if d > dim {
            d = 1;
            m += 1;
            if m > 12 {
                m = 1;
                y += 1;
            }
        }
    }
    out
}

struct Gen<'a> {
    rng: &'a mut StdRng,
    dates: &'a [i32],
}

impl Gen<'_> {
    /// Bars from base returns `b`, extra intraday drift `extra`, log-volume shift `vshift`.
    fn bars(
        &mut self,
        symbol: &str,
        b: &[f64],
        extra: &[f64],
        vshift: &[f64],
        adj: f64,
        keep: &[bool],
    ) -> Bars {
        let n = b.len();
        let mut out = Bars {
            symbol: symbol.into(),
            date: Vec::with_capacity(n),
            open: Vec::with_capacity(n),
            high: Vec::with_capacity(n),
            low: Vec::with_capacity(n),
            close: Vec::with_capacity(n),
            adj_close: Vec::with_capacity(n),
            volume: Vec::with_capacity(n),
        };
        let mut c_prev = 50.0 * (0.3 * normal(self.rng)).exp();
        let v0 = 10f64.powf(6.0 + self.rng.random::<f64>());
        for k in 0..n {
            let gap = 0.25 * b[k] + 0.002 * normal(self.rng);
            let intra = b[k] - gap + extra[k];
            let o = c_prev * gap.exp();
            let c = o * intra.exp();
            let h = o.max(c) * (0.004 * normal(self.rng)).abs().exp();
            let l = o.min(c) * (-(0.004 * normal(self.rng)).abs()).exp();
            let v = v0 * (0.3 * normal(self.rng) + vshift[k]).exp();
            c_prev = c;
            if !keep[k] {
                continue;
            }
            out.date.push(self.dates[k]);
            out.open.push(o);
            out.high.push(h);
            out.low.push(l);
            out.close.push(c);
            out.adj_close.push(c * adj);
            out.volume.push(v);
        }
        out
    }
}

/// Generates a synthetic panel (deterministic in `config`).
pub fn generate(config: &SyntheticConfig) -> SyntheticPanel {
    let n = config.num_days;
    let mut rng = StdRng::seed_from_u64(config.seed);
    let dates = weekday_calendar(n);
    let zeros = vec![0.0; n];
    let all = vec![true; n];
    let (drift, loading) = config.signal.map_or((0.0, 0.0), |p| (p.drift, p.loading));

    let m: Vec<f64> = (0..n).map(|_| 0.0003 + 0.01 * normal(&mut rng)).collect();
    let mut sec_ret = Vec::new();
    for _ in 0..config.num_sectors {
        let beta = 0.8 + 0.4 * rng.random::<f64>();
        sec_ret.push(
            (0..n)
                .map(|k| beta * m[k] + 0.006 * normal(&mut rng))
                .collect::<Vec<_>>(),
        );
    }
    let ns = config.num_sectors * config.stocks_per_sector;
    let mut regime = vec![0i8; ns * n];
    let mut stock_inputs = Vec::new();
    let mut stock_sector = Vec::new();
    for s in 0..ns {
        let j = s / config.stocks_per_sector;
        let beta = 0.7 + 0.6 * rng.random::<f64>();
        let b: Vec<f64> = (0..n)
            .map(|k| beta * sec_ret[j][k] + 0.012 * normal(&mut rng))
            .collect();
        let z: Vec<f64> = (0..n)
            .map(|_| if rng.random::<bool>() { 1.0 } else { -1.0 })
            .collect();
        for (k, zk) in z.iter().enumerate() {
            regime[s * n + k] = *zk as i8;
        }
        let extra: Vec<f64> = (0..n)
            .map(|k| if k == 0 { 0.0 } else { drift * z[k - 1] })
            .collect();
        let vshift: Vec<f64> = z.iter().map(|zk| loading * zk).collect();
        let keep: Vec<bool> = (0..n)
            .map(|_| rng.random::<f64>() >= config.missing_prob)
            .collect();
        stock_inputs.push((b, extra, vshift, keep));
        stock_sector.push(j);
    }

    let mut g = Gen {
        rng: &mut rng,
        dates: &dates,
    };
    let market = g.bars("MKT", &m, &zeros, &zeros, 1.0, &all);
    let sectors: Vec<Bars> = sec_ret
        .iter()
        .enumerate()
        .map(|(j, r)| g.bars(&format!("SEC{j}"), r, &zeros, &zeros, 1.0, &all))
        .collect();
    let stocks: Vec<Bars> = stock_inputs
        .iter()
        .enumerate()
        .map(|(s, (b, e, v, keep))| g.bars(&format!("S{s:03}"), b, e, v, 0.9, keep))
        .collect();
    let (panel, _) = Panel::from_bars(&market, &sectors, &stocks, &stock_sector)
        .expect("synthetic bars are valid");
    let mut bars = vec![market];
    bars.extend(sectors);
    bars.extend(stocks);
    SyntheticPanel {
        panel,
        regime,
        bars,
    }
}
