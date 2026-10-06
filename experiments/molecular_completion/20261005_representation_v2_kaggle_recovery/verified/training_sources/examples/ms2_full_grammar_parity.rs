//! Check the Python completion experiment against the authoritative Rust grammar.
use std::io::{self, Read};
use mamba3::models::ms2::{Composition, Limits, Token, TraceState, atom_type, HYDROGEN, STOP};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Case { counts: [u16; 4], trace: Vec<[u8; 4]> }
#[derive(Serialize)]
struct Report { masks: Vec<[u32; 4]>, residual: Vec<u8> }

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input)?;
    let cases: Vec<Case> = serde_json::from_str(&input)?;
    let mut reports = Vec::new();
    for case in cases {
        let mut budget: Composition = [0; 10];
        // Both composition tables begin C,H,N,O.
        budget[0] = case.counts[0];
        budget[HYDROGEN] = case.counts[1];
        budget[2] = case.counts[2];
        budget[3] = case.counts[3];
        let mut state = TraceState::new(Limits::new(32, 8)?, Some(budget));
        let mut used: Composition = [0; 10];
        let mut masks = Vec::new();
        for fields in case.trace {
            let token = Token { kind: fields[0], atom_type: fields[1], bond: fields[2], pointer: fields[3] };
            let legal = state.masks(token);
            let mut kinds = legal.kinds;
            if used != budget || state.residual_valence().iter().any(|r| *r != 0) {
                kinds &= !(1u32 << STOP);
            }
            if kinds & (1u32 << token.kind) == 0 {
                return Err("incomplete STOP or illegal kind".into());
            }
            masks.push([kinds, legal.atom_types, legal.bonds, legal.pointers]);
            state.apply(token)?;
            if token.kind == mamba3::models::ms2::ADD_ATOM {
                let atom = atom_type(token.atom_type).ok_or("unknown atom type")?;
                used[atom.element] += 1;
                used[HYDROGEN] += u16::from(atom.hydrogens);
            }
        }
        if used != budget || state.residual_valence().iter().any(|r| *r != 0) {
            return Err("completed graph is not closed under exact composition".into());
        }
        reports.push(Report { masks, residual: state.residual_valence().to_vec() });
    }
    println!("{}", serde_json::to_string(&reports)?);
    Ok(())
}
