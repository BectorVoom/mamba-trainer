//! Learning-rate schedules.

/// A learning-rate schedule evaluated per optimizer step.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum LrSchedule {
    /// Hold the base rate.
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

impl Default for LrSchedule {
    fn default() -> Self {
        LrSchedule::Constant
    }
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
                let decays = if every == 0 { 0 } else { step / every };
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
}
