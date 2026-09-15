// Whole-sentence correction: candidate generation, beam search, display policy.
// Mirrors crates/boshiamy_core (candidate_generator.rs, sentence_ranker.rs,
// correction_policy.rs). Keep the numbers in sync with ScoringWeights::mvp_defaults()
// and CorrectionPolicyConfig::mvp_defaults().

import { CodeIndex, isCjk } from './cin';
import { NgramModel, BOS } from './lm';

export interface SessionUnit {
  ch: string;
  rawCode: string;
  /** > 0 when the user explicitly picked a non-first candidate. */
  selectedIndex: number;
}

export interface Weights {
  lambdaEdit: number;
  lambdaChange: number;
  lambdaChoice: number;
  beamWidth: number;
  maxCandidatesPerPosition: number;
  minScoreDelta: number;
  maxChangedAbsolute: number;
  maxChangedFraction: number;
}

export const DEFAULT_WEIGHTS: Weights = {
  lambdaEdit: 3.0,
  lambdaChange: 3.0,
  lambdaChoice: 6.0,
  beamWidth: 32,
  maxCandidatesPerPosition: 32,
  minScoreDelta: 4.0,
  maxChangedAbsolute: 3,
  maxChangedFraction: 0.25,
};

export interface Suggestion {
  original: string;
  corrected: string;
  /** Indices (in chars) that differ. */
  changed: number[];
  score: number;
  originalScore: number;
  changedCount: number;
  elapsedMs: number;
}

interface Candidate {
  ch: string;
  id: number;
  distance: number;
  isOriginal: boolean;
}

interface Beam {
  ids: number[];
  chars: string[];
  distance: number;
  changed: number;
  changedExplicit: number;
  score: number;
}

export class CorrectionEngine {
  constructor(
    public index: CodeIndex,
    public lm: NgramModel,
    public weights: Weights = DEFAULT_WEIGHTS,
  ) {}

  private candidates(unit: SessionUnit): Candidate[] {
    const original: Candidate = {
      ch: unit.ch,
      id: this.lm.id(unit.ch),
      distance: 0,
      isOriginal: true,
    };
    if (!unit.rawCode || !isCjk(unit.ch)) return [original];
    const byChar = new Map<string, Candidate>([[unit.ch, original]]);
    const add = (ch: string, distance: number) => {
      if (!isCjk(ch)) return;
      const existing = byChar.get(ch);
      if (existing) {
        if (distance < existing.distance) existing.distance = distance;
      } else byChar.set(ch, { ch, id: this.lm.id(ch), distance, isOriginal: false });
    };
    for (const ch of this.index.charsForCode(unit.rawCode)) add(ch, 0);
    for (const [, ch] of this.index.substitutionNeighbors(unit.rawCode)) add(ch, 1);
    const list = Array.from(byChar.values()).map((c) => ({ c, prior: this.lm.unigram(c.ch) }));
    list.sort((x, y) => {
      if (x.c.isOriginal !== y.c.isOriginal) return x.c.isOriginal ? -1 : 1;
      if (x.c.distance !== y.c.distance) return x.c.distance - y.c.distance;
      if (x.prior !== y.prior) return y.prior - x.prior;
      return x.c.ch < y.c.ch ? -1 : 1;
    });
    return list.slice(0, this.weights.maxCandidatesPerPosition).map((x) => x.c);
  }

  private maxAllowedChanges(len: number): number {
    if (len === 0) return 0;
    const byFrac = Math.floor(len * this.weights.maxChangedFraction);
    return Math.max(Math.min(byFrac, this.weights.maxChangedAbsolute), Math.min(1, len));
  }

  suggest(units: SessionUnit[]): Suggestion | null {
    const t0 = Date.now();
    if (units.length === 0) return null;
    const original = units.map((u) => u.ch).join('');
    if (shouldSkipText(original)) return null;
    const w = this.weights;
    const lm = this.lm;

    let beam: Beam[] = [{ ids: [], chars: [], distance: 0, changed: 0, changedExplicit: 0, score: 0 }];
    for (const unit of units) {
      const cands = this.candidates(unit);
      const next: Beam[] = [];
      for (const state of beam) {
        const n = state.ids.length;
        const a = n >= 2 ? state.ids[n - 2] : BOS;
        const b = n >= 1 ? state.ids[n - 1] : BOS;
        for (const cand of cands) {
          const changed = cand.ch !== unit.ch;
          const explicit = changed && unit.selectedIndex > 0;
          const penalty =
            w.lambdaEdit * cand.distance + (changed ? w.lambdaChange : 0) + (explicit ? w.lambdaChoice : 0);
          next.push({
            ids: [...state.ids, cand.id],
            chars: [...state.chars, cand.ch],
            distance: state.distance + cand.distance,
            changed: state.changed + (changed ? 1 : 0),
            changedExplicit: state.changedExplicit + (explicit ? 1 : 0),
            score: state.score + lm.logprob(a, b, cand.id) - penalty,
          });
        }
      }
      next.sort((x, y) => y.score - x.score);
      beam = next.slice(0, Math.max(1, w.beamWidth));
    }

    const originalScore = lm.scoreSentence(original);
    const maxChanges = this.maxAllowedChanges(units.length);
    const ranked = beam
      .map((s) => {
        const n = s.ids.length;
        const a = n >= 2 ? s.ids[n - 2] : BOS;
        const b = n >= 1 ? s.ids[n - 1] : BOS;
        // Add </s> so the full-sentence score matches lm.scoreSentence.
        const full = s.score + lm.logprob(a, b, 1) + w.lambdaEdit * s.distance + w.lambdaChange * s.changed + w.lambdaChoice * s.changedExplicit;
        const score = full - w.lambdaEdit * s.distance - w.lambdaChange * s.changed - w.lambdaChoice * s.changedExplicit;
        return { ...s, score };
      })
      .sort((x, y) => y.score - x.score);

    for (const cand of ranked) {
      const text = cand.chars.join('');
      if (text === original || cand.changed === 0 || cand.changed > maxChanges) continue;
      const delta = cand.score - originalScore;
      if (delta < w.minScoreDelta) continue;
      const changed: number[] = [];
      cand.chars.forEach((c, i) => {
        if (c !== units[i].ch) changed.push(i);
      });
      return {
        original,
        corrected: text,
        changed,
        score: cand.score,
        originalScore,
        changedCount: cand.changed,
        elapsedMs: Date.now() - t0,
      };
    }
    return null;
  }
}

function shouldSkipText(text: string): boolean {
  const t = text.trim();
  if (!t) return true;
  const lower = t.toLowerCase();
  if (lower.startsWith('http://') || lower.startsWith('https://') || lower.startsWith('www.') || lower.includes('://')) return true;
  const at = t.indexOf('@');
  if (at > 0 && at < t.length - 1 && t.slice(at + 1).includes('.')) return true;
  const chars = Array.from(t).filter((c) => !/\s/.test(c));
  if (chars.length === 0) return false;
  const cjk = chars.filter(isCjk).length;
  if (cjk === 0) return true;
  const digits = chars.filter((c) => /[0-9]/.test(c)).length;
  return digits / chars.length >= 0.7;
}
