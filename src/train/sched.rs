//! Learning-rate schedules.

use crate::error::{Error, Result};

/// A learning-rate schedule evaluated per optimizer step.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize, Default)]
pub enum LrSchedule {
    /// Hold the base rate.
    #[default]
    Constant,
    /// Linear warmup, then cosine decay to `min_lr`.
    CosineWithWarmup {
        /// Steps spent ramping up from zero.
        warmup_steps: u64,
        /// Total steps in the run.
        total_steps: u64,
        /// Floor the cosine decays to, as a fraction of the base rate.
        min_ratio: f32,
    },
    /// Linear warmup, then linear decay to `min_ratio`.
    LinearWithWarmup {
        /// Steps spent ramping up from zero.
        warmup_steps: u64,
        /// Total steps in the run.
        total_steps: u64,
        /// Floor as a fraction of the base rate.
        min_ratio: f32,
    },
    /// Multiply by `gamma` every `every` steps.
    Step {
        /// Steps between decays.
        every: u64,
        /// Decay factor.
        gamma: f32,
    },
    /// `base / sqrt(max(step, warmup))`, the Transformer schedule.
    InverseSqrt {
        /// Warmup length.
        warmup_steps: u64,
    },
}

impl LrSchedule {
    /// The learning rate at `step` (1-based), given the base rate.
    pub fn at(&self, base: f32, step: u64) -> f32 {
        match *self {
            LrSchedule::Constant => base,
            LrSchedule::CosineWithWarmup {
                warmup_steps,
                total_steps,
                min_ratio,
            } => {
                if step < warmup_steps {
                    return base * (step as f32 / warmup_steps.max(1) as f32);
                }
                let span = total_steps.saturating_sub(warmup_steps).max(1) as f32;
                let progress = ((step - warmup_steps) as f32 / span).clamp(0.0, 1.0);
                let cosine = 0.5 * (1.0 + (core::f32::consts::PI * progress).cos());
                base * (min_ratio + (1.0 - min_ratio) * cosine)
            }
            LrSchedule::LinearWithWarmup {
                warmup_steps,
                total_steps,
                min_ratio,
            } => {
                if step < warmup_steps {
                    return base * (step as f32 / warmup_steps.max(1) as f32);
                }
                let span = total_steps.saturating_sub(warmup_steps).max(1) as f32;
                let progress = ((step - warmup_steps) as f32 / span).clamp(0.0, 1.0);
                base * (min_ratio + (1.0 - min_ratio) * (1.0 - progress))
            }
            LrSchedule::Step { every, gamma } => {
                let decays = step.checked_div(every).unwrap_or(0);
                base * gamma.powi(decays as i32)
            }
            LrSchedule::InverseSqrt { warmup_steps } => {
                let s = step.max(1) as f32;
                let w = warmup_steps.max(1) as f32;
                if step < warmup_steps {
                    base * (s / w)
                } else {
                    base * (w / s).sqrt()
                }
            }
        }
    }

    /// A cosine schedule covering `total_steps` with a 2% warmup.
    pub fn cosine(total_steps: u64) -> Self {
        LrSchedule::CosineWithWarmup {
            warmup_steps: (total_steps / 50).max(1),
            total_steps,
            min_ratio: 0.1,
        }
    }

    /// Check internal consistency: finite rates and ratios, a nonzero length
    /// where the schedule has one, and a warmup that fits inside it.
    pub fn validate(&self) -> Result<()> {
        match *self {
            LrSchedule::Constant => Ok(()),
            LrSchedule::CosineWithWarmup {
                warmup_steps,
                total_steps,
                min_ratio,
            }
            | LrSchedule::LinearWithWarmup {
                warmup_steps,
                total_steps,
                min_ratio,
            } => {
                if total_steps == 0 {
                    return Err(Error::config(
                        "a schedule needs at least one total step".to_string(),
                    ));
                }
                if warmup_steps > total_steps {
                    return Err(Error::config(format!(
                        "warmup_steps ({warmup_steps}) cannot exceed total_steps ({total_steps})"
                    )));
                }
                if !min_ratio.is_finite() || min_ratio < 0.0 {
                    return Err(Error::config(format!(
                        "min_ratio must be a nonnegative finite fraction of the base \
                         rate, got {min_ratio}"
                    )));
                }
                Ok(())
            }
            LrSchedule::Step { every, gamma } => {
                if every == 0 {
                    return Err(Error::config(
                        "every must be positive, or the schedule never decays".to_string(),
                    ));
                }
                if !gamma.is_finite() || gamma < 0.0 {
                    return Err(Error::config(format!(
                        "gamma must be a nonnegative finite decay factor, got {gamma}"
                    )));
                }
                Ok(())
            }
            LrSchedule::InverseSqrt { warmup_steps } => {
                if warmup_steps == 0 {
                    return Err(Error::config("warmup_steps must be positive".to_string()));
                }
                Ok(())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_warms_up_then_decays() {
        let s = LrSchedule::CosineWithWarmup {
            warmup_steps: 10,
            total_steps: 110,
            min_ratio: 0.0,
        };
        assert!((s.at(1.0, 0) - 0.0).abs() < 1e-6);
        assert!((s.at(1.0, 5) - 0.5).abs() < 1e-6);
        assert!((s.at(1.0, 10) - 1.0).abs() < 1e-6);
        assert!((s.at(1.0, 110) - 0.0).abs() < 1e-6);
        // Monotone decay after warmup.
        assert!(s.at(1.0, 40) > s.at(1.0, 80));
    }

    #[test]
    fn step_schedule_halves() {
        let s = LrSchedule::Step {
            every: 10,
            gamma: 0.5,
        };
        assert!((s.at(1.0, 9) - 1.0).abs() < 1e-6);
        assert!((s.at(1.0, 10) - 0.5).abs() < 1e-6);
        assert!((s.at(1.0, 25) - 0.25).abs() < 1e-6);
    }

    #[test]
    fn constant_always_validates() {
        LrSchedule::Constant
            .validate()
            .expect("the default is always legal");
    }

    #[test]
    fn a_warmup_longer_than_the_run_is_refused() {
        let s = LrSchedule::CosineWithWarmup {
            warmup_steps: 20,
            total_steps: 10,
            min_ratio: 0.1,
        };
        assert!(s.validate().is_err());
    }

    #[test]
    fn a_zero_length_schedule_is_refused() {
        assert!(
            LrSchedule::CosineWithWarmup {
                warmup_steps: 0,
                total_steps: 0,
                min_ratio: 0.1,
            }
            .validate()
            .is_err()
        );
        assert!(
            LrSchedule::LinearWithWarmup {
                warmup_steps: 0,
                total_steps: 0,
                min_ratio: 0.0,
            }
            .validate()
            .is_err()
        );
    }

    #[test]
    fn a_nonfinite_or_negative_ratio_is_refused() {
        for min_ratio in [f32::NAN, f32::INFINITY, -0.1] {
            assert!(
                LrSchedule::CosineWithWarmup {
                    warmup_steps: 1,
                    total_steps: 10,
                    min_ratio,
                }
                .validate()
                .is_err(),
                "min_ratio {min_ratio} should have been refused"
            );
        }
    }

    #[test]
    fn a_zero_step_interval_is_refused() {
        assert!(
            LrSchedule::Step {
                every: 0,
                gamma: 0.5
            }
            .validate()
            .is_err(),
            "every=0 would never decay, silently acting like Constant"
        );
    }

    #[test]
    fn a_nonfinite_or_negative_gamma_is_refused() {
        for gamma in [f32::NAN, f32::INFINITY, -1.0] {
            assert!(
                LrSchedule::Step { every: 10, gamma }.validate().is_err(),
                "gamma {gamma} should have been refused"
            );
        }
    }

    #[test]
    fn zero_inverse_sqrt_warmup_is_refused() {
        assert!(
            LrSchedule::InverseSqrt { warmup_steps: 0 }
                .validate()
                .is_err()
        );
    }
}
