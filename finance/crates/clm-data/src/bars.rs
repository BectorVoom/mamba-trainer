//! Daily bars (CONTRACTS C1): one symbol's Parquet file, and the day-aligned panel of
//! stocks, market index and sector ETFs on the market's calendar.

use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, Float64Array, Int32Array, RecordBatch};
use arrow::compute::cast;
use arrow::datatypes::DataType;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use serde::{Deserialize, Serialize};

use crate::error::{DataError, Result};

/// One symbol's raw bars as stored on disk (unadjusted O/H/L/C, `adj_close`, volume).
#[derive(Debug, Clone, PartialEq)]
pub struct Bars {
    /// Ticker.
    pub symbol: String,
    /// `yyyymmdd`, strictly increasing.
    pub date: Vec<i32>,
    /// Open as reported (split-adjusted, not dividend-adjusted).
    pub open: Vec<f64>,
    /// High as reported.
    pub high: Vec<f64>,
    /// Low as reported.
    pub low: Vec<f64>,
    /// Close as reported.
    pub close: Vec<f64>,
    /// Close adjusted for splits and dividends.
    pub adj_close: Vec<f64>,
    /// Volume (shares).
    pub volume: Vec<f64>,
}

impl Bars {
    /// Number of rows.
    pub fn len(&self) -> usize {
        self.date.len()
    }

    /// True if there are no rows.
    pub fn is_empty(&self) -> bool {
        self.date.is_empty()
    }

    /// Checks column lengths and that dates are strictly increasing.
    pub fn validate(&self) -> Result<()> {
        let n = self.date.len();
        for (name, col) in [
            ("open", &self.open),
            ("high", &self.high),
            ("low", &self.low),
            ("close", &self.close),
            ("adj_close", &self.adj_close),
            ("volume", &self.volume),
        ] {
            if col.len() != n {
                return Err(DataError::Invalid(format!(
                    "{}: column {name} has {} rows, date has {n}",
                    self.symbol,
                    col.len()
                )));
            }
        }
        if let Some(i) = (1..n).find(|&i| self.date[i] <= self.date[i - 1]) {
            return Err(DataError::Invalid(format!(
                "{}: dates not strictly increasing at row {i} ({} then {})",
                self.symbol,
                self.date[i - 1],
                self.date[i]
            )));
        }
        Ok(())
    }
}

/// Reads one symbol's bars from Parquet (columns `date` int32 `yyyymmdd`, `open`,
/// `high`, `low`, `close`, `adj_close`, `volume` float64; other integer/float types are
/// cast). The symbol is the file stem. Validates column lengths and date order.
pub fn load_parquet(path: impl AsRef<Path>) -> Result<Bars> {
    let path = path.as_ref();
    let symbol = path
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or_else(|| DataError::Invalid(format!("bad file name {}", path.display())))?
        .to_string();
    let file = File::open(path).map_err(|e| DataError::io(path, e))?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)?.build()?;
    let mut bars = Bars {
        symbol,
        date: Vec::new(),
        open: Vec::new(),
        high: Vec::new(),
        low: Vec::new(),
        close: Vec::new(),
        adj_close: Vec::new(),
        volume: Vec::new(),
    };
    for batch in reader {
        let batch = batch?;
        bars.date.extend(i32_column(&batch, "date")?);
        bars.open.extend(f64_column(&batch, "open")?);
        bars.high.extend(f64_column(&batch, "high")?);
        bars.low.extend(f64_column(&batch, "low")?);
        bars.close.extend(f64_column(&batch, "close")?);
        bars.adj_close.extend(f64_column(&batch, "adj_close")?);
        bars.volume.extend(f64_column(&batch, "volume")?);
    }
    bars.validate()?;
    Ok(bars)
}

pub(crate) fn column(batch: &RecordBatch, name: &str) -> Result<ArrayRef> {
    batch
        .column_by_name(name)
        .cloned()
        .ok_or_else(|| DataError::Schema(format!("missing column {name}")))
}

fn i32_column(batch: &RecordBatch, name: &str) -> Result<Vec<i32>> {
    let col = cast(&column(batch, name)?, &DataType::Int32)?;
    let arr = col
        .as_any()
        .downcast_ref::<Int32Array>()
        .ok_or_else(|| DataError::Schema(format!("{name}: not int32")))?;
    if arr.null_count() > 0 {
        return Err(DataError::Schema(format!("{name}: has nulls")));
    }
    Ok(arr.values().to_vec())
}

/// Float column; nulls become NaN (and therefore a missing bar).
pub(crate) fn f64_column(batch: &RecordBatch, name: &str) -> Result<Vec<f64>> {
    let col = cast(&column(batch, name)?, &DataType::Float64)?;
    let arr = col
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| DataError::Schema(format!("{name}: not float64")))?;
    Ok((0..arr.len())
        .map(|i| {
            if arr.is_null(i) {
                f64::NAN
            } else {
                arr.value(i)
            }
        })
        .collect())
}

/// One security's bars aligned to the panel calendar, **adjusted** per C1:
/// `f = adj_close / close`, `O = open*f`, `H = high*f`, `L = low*f`, `C = adj_close`.
/// A missing day has `present = false` and NaN in every column. Equality is bitwise
/// (so two series with the same missing days compare equal despite the NaNs).
#[derive(Debug, Clone)]
pub struct Series {
    /// Ticker.
    pub symbol: String,
    /// Adjusted open.
    pub open: Vec<f64>,
    /// Adjusted high.
    pub high: Vec<f64>,
    /// Adjusted low.
    pub low: Vec<f64>,
    /// Adjusted close (`adj_close`).
    pub close: Vec<f64>,
    /// Volume (shares, as reported; `>= 0`).
    pub volume: Vec<f64>,
    /// Traded value on the day, `close * volume` with the **reported** (unadjusted for
    /// dividends) close. Point-in-time, unlike `adj_close * volume` (see `features`).
    pub dollar_volume: Vec<f64>,
    /// Whether the security has a bar on the day.
    pub present: Vec<bool>,
}

impl PartialEq for Series {
    fn eq(&self, other: &Self) -> bool {
        let bits = |a: &[f64], b: &[f64]| {
            a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
        };
        self.symbol == other.symbol
            && self.present == other.present
            && bits(&self.open, &other.open)
            && bits(&self.high, &other.high)
            && bits(&self.low, &other.low)
            && bits(&self.close, &other.close)
            && bits(&self.volume, &other.volume)
            && bits(&self.dollar_volume, &other.dollar_volume)
    }
}

impl Series {
    /// A series of `n` missing days.
    pub fn missing(symbol: &str, n: usize) -> Self {
        Self {
            symbol: symbol.to_string(),
            open: vec![f64::NAN; n],
            high: vec![f64::NAN; n],
            low: vec![f64::NAN; n],
            close: vec![f64::NAN; n],
            volume: vec![f64::NAN; n],
            dollar_volume: vec![f64::NAN; n],
            present: vec![false; n],
        }
    }

    /// Number of calendar days.
    pub fn len(&self) -> usize {
        self.present.len()
    }

    /// True if the calendar is empty.
    pub fn is_empty(&self) -> bool {
        self.present.is_empty()
    }

    /// True if the bar of day `t` exists (`false` outside the calendar).
    #[inline]
    pub fn has(&self, t: usize) -> bool {
        self.present.get(t).copied().unwrap_or(false)
    }

    /// True if every day in `lo..=hi` has a bar (`false` if `hi` is past the calendar).
    #[inline]
    pub fn has_all(&self, lo: usize, hi: usize) -> bool {
        hi < self.len() && (lo..=hi).all(|k| self.present[k])
    }

    /// Marks day `t` missing.
    pub fn set_missing(&mut self, t: usize) {
        self.present[t] = false;
        self.open[t] = f64::NAN;
        self.high[t] = f64::NAN;
        self.low[t] = f64::NAN;
        self.close[t] = f64::NAN;
        self.volume[t] = f64::NAN;
        self.dollar_volume[t] = f64::NAN;
    }

    /// The first `n` days.
    pub fn truncated(&self, n: usize) -> Self {
        Self {
            symbol: self.symbol.clone(),
            open: self.open[..n].to_vec(),
            high: self.high[..n].to_vec(),
            low: self.low[..n].to_vec(),
            close: self.close[..n].to_vec(),
            volume: self.volume[..n].to_vec(),
            dollar_volume: self.dollar_volume[..n].to_vec(),
            present: self.present[..n].to_vec(),
        }
    }
}

/// What happened to one symbol's rows while aligning it to the calendar.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SymbolReport {
    /// Ticker.
    pub symbol: String,
    /// Rows in the file.
    pub rows: usize,
    /// Rows kept as bars.
    pub kept: usize,
    /// Dropped: an O/H/L/C/adj_close value non-positive or not finite.
    pub dropped_bad_price: usize,
    /// Dropped: volume negative or not finite.
    pub dropped_bad_volume: usize,
    /// Dropped: date not on the calendar.
    pub dropped_off_calendar: usize,
    /// Kept, but `high < low` or open/close outside `[low, high]` (relative tolerance 1e-6).
    pub inconsistent_ohlc: usize,
    /// Kept bars with zero volume (volume features of the rows reading them are invalid).
    pub zero_volume: usize,
    /// Calendar days without a bar.
    pub missing_days: usize,
}

/// Per-symbol load report (market first, then sector ETFs, then stocks).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LoadReport {
    /// One entry per symbol.
    pub symbols: Vec<SymbolReport>,
}

impl LoadReport {
    /// Total rows dropped for bad prices or volume.
    pub fn total_dropped_bad(&self) -> usize {
        self.symbols
            .iter()
            .map(|r| r.dropped_bad_price + r.dropped_bad_volume)
            .sum()
    }

    /// Total rows dropped because their date is not on the calendar.
    pub fn total_off_calendar(&self) -> usize {
        self.symbols.iter().map(|r| r.dropped_off_calendar).sum()
    }
}

/// `universe.json`: market index and sector ETF → stocks map (other keys ignored).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Universe {
    /// Market index symbol (its trading days are the calendar).
    pub market: String,
    /// Sector ETF → member stocks.
    pub sectors: BTreeMap<String, Vec<String>>,
}

/// Every security on one common calendar (the market index's dates, C1).
///
/// Stocks are sorted by symbol, sector ETFs by symbol; `sector_of[s]` indexes `sectors`.
#[derive(Debug, Clone, PartialEq)]
pub struct Panel {
    /// Calendar, `yyyymmdd`, strictly increasing.
    pub dates: Vec<i32>,
    /// Market index.
    pub market: Series,
    /// Sector ETFs.
    pub sectors: Vec<Series>,
    /// Stocks (the securities that get features, labels and samples).
    pub stocks: Vec<Series>,
    /// Sector ETF index of each stock.
    pub sector_of: Vec<usize>,
}

impl Panel {
    /// Number of calendar days.
    pub fn num_days(&self) -> usize {
        self.dates.len()
    }

    /// Number of stocks.
    pub fn num_securities(&self) -> usize {
        self.stocks.len()
    }

    /// Stock symbols in index order.
    pub fn symbols(&self) -> Vec<String> {
        self.stocks.iter().map(|s| s.symbol.clone()).collect()
    }

    /// The sector ETF series of stock `s`.
    pub fn sector(&self, s: usize) -> &Series {
        &self.sectors[self.sector_of[s]]
    }

    /// Calendar year of day `t`.
    pub fn year(&self, t: usize) -> i32 {
        self.dates[t] / 10_000
    }

    /// Calendar year of every day (input to fold construction, C4).
    pub fn years(&self) -> Vec<i32> {
        self.dates.iter().map(|d| d / 10_000).collect()
    }

    /// The panel of days `0..=last` only (what was known after the close of `last`).
    pub fn truncated(&self, last: usize) -> Self {
        let n = last + 1;
        assert!(n <= self.num_days(), "truncate past the calendar");
        Self {
            dates: self.dates[..n].to_vec(),
            market: self.market.truncated(n),
            sectors: self.sectors.iter().map(|s| s.truncated(n)).collect(),
            stocks: self.stocks.iter().map(|s| s.truncated(n)).collect(),
            sector_of: self.sector_of.clone(),
        }
    }

    /// Builds a panel from raw bars. The calendar is `market.date`; `stock_sector[i]`
    /// is the index into `sectors` of `stocks[i]`. Stocks keep the given order.
    /// Rows with a non-positive or non-finite price or a negative/non-finite volume, and
    /// rows off the calendar, are dropped and counted in the report.
    pub fn from_bars(
        market: &Bars,
        sectors: &[Bars],
        stocks: &[Bars],
        stock_sector: &[usize],
    ) -> Result<(Self, LoadReport)> {
        if stocks.len() != stock_sector.len() {
            return Err(DataError::Invalid("one sector index per stock".into()));
        }
        if let Some(&bad) = stock_sector.iter().find(|&&k| k >= sectors.len()) {
            return Err(DataError::Invalid(format!(
                "sector index {bad} out of range"
            )));
        }
        market.validate()?;
        let dates = market.date.clone();
        let index: HashMap<i32, usize> = dates.iter().enumerate().map(|(i, &d)| (d, i)).collect();
        let mut report = LoadReport::default();
        let mut align = |bars: &Bars| -> Result<Series> {
            let (series, rep) = align_bars(bars, &index, dates.len())?;
            report.symbols.push(rep);
            Ok(series)
        };
        let market_series = align(market)?;
        let sector_series = sectors.iter().map(&mut align).collect::<Result<Vec<_>>>()?;
        let stock_series = stocks.iter().map(&mut align).collect::<Result<Vec<_>>>()?;
        Ok((
            Self {
                dates,
                market: market_series,
                sectors: sector_series,
                stocks: stock_series,
                sector_of: stock_sector.to_vec(),
            },
            report,
        ))
    }

    /// Loads `dir/universe.json` and `dir/<SYMBOL>.parquet` for the market, every sector
    /// ETF and every stock. Stocks and sectors are sorted by symbol.
    pub fn load_dir(dir: impl AsRef<Path>) -> Result<(Self, LoadReport)> {
        let dir = dir.as_ref();
        let upath = dir.join("universe.json");
        let text = std::fs::read_to_string(&upath).map_err(|e| DataError::io(&upath, e))?;
        let universe: Universe = serde_json::from_str(&text)?;
        let load = |sym: &str| load_parquet(dir.join(format!("{sym}.parquet")));
        let market = load(&universe.market)?;
        let sector_syms: Vec<&String> = universe.sectors.keys().collect();
        let sectors = sector_syms
            .iter()
            .map(|s| load(s))
            .collect::<Result<Vec<_>>>()?;
        let mut members: Vec<(String, usize)> = Vec::new();
        for (k, etf) in sector_syms.iter().enumerate() {
            for stock in &universe.sectors[*etf] {
                if members.iter().any(|(s, _)| s == stock) {
                    return Err(DataError::Invalid(format!("{stock} is in two sectors")));
                }
                members.push((stock.clone(), k));
            }
        }
        members.sort();
        let stocks = members
            .iter()
            .map(|(s, _)| load(s))
            .collect::<Result<Vec<_>>>()?;
        let stock_sector: Vec<usize> = members.iter().map(|&(_, k)| k).collect();
        Self::from_bars(&market, &sectors, &stocks, &stock_sector)
    }
}

fn align_bars(
    bars: &Bars,
    index: &HashMap<i32, usize>,
    n: usize,
) -> Result<(Series, SymbolReport)> {
    bars.validate()?;
    let mut series = Series::missing(&bars.symbol, n);
    let mut rep = SymbolReport {
        symbol: bars.symbol.clone(),
        rows: bars.len(),
        ..Default::default()
    };
    let good = |x: f64| x.is_finite() && x > 0.0;
    for i in 0..bars.len() {
        let Some(&t) = index.get(&bars.date[i]) else {
            rep.dropped_off_calendar += 1;
            continue;
        };
        let (o, h, l, c, a, v) = (
            bars.open[i],
            bars.high[i],
            bars.low[i],
            bars.close[i],
            bars.adj_close[i],
            bars.volume[i],
        );
        if ![o, h, l, c, a].into_iter().all(good) {
            rep.dropped_bad_price += 1;
            continue;
        }
        if !(v.is_finite() && v >= 0.0) {
            rep.dropped_bad_volume += 1;
            continue;
        }
        let tol = 1e-6 * h;
        if h < l || o > h + tol || c > h + tol || o < l - tol || c < l - tol {
            rep.inconsistent_ohlc += 1;
        }
        if v == 0.0 {
            rep.zero_volume += 1;
        }
        let f = a / c;
        series.open[t] = o * f;
        series.high[t] = h * f;
        series.low[t] = l * f;
        series.close[t] = a;
        series.volume[t] = v;
        series.dollar_volume[t] = c * v;
        series.present[t] = true;
        rep.kept += 1;
    }
    rep.missing_days = n - rep.kept;
    Ok((series, rep))
}

/// Writes bars in the on-disk layout `load_parquet` reads (for tools and tests).
pub fn write_parquet(bars: &Bars, path: impl AsRef<Path>) -> Result<()> {
    bars.validate()?;
    let f64_array = |v: &[f64]| -> ArrayRef { Arc::new(Float64Array::from(v.to_vec())) };
    let batch = RecordBatch::try_from_iter([
        (
            "date",
            Arc::new(Int32Array::from(bars.date.clone())) as ArrayRef,
        ),
        ("open", f64_array(&bars.open)),
        ("high", f64_array(&bars.high)),
        ("low", f64_array(&bars.low)),
        ("close", f64_array(&bars.close)),
        ("adj_close", f64_array(&bars.adj_close)),
        ("volume", f64_array(&bars.volume)),
    ])?;
    crate::export::write_batch(path.as_ref(), &batch)
}
