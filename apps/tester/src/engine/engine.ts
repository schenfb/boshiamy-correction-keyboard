// Whole-sentence correction: lattice decode + display policy.
// Mirrors crates/boshiamy_core (lib.rs, lattice.rs, correction_policy.rs). Keep
// the numbers in sync with LatticeWeights::mvp_defaults() and
// CorrectionPolicyConfig::mvp_defaults().

import { CodeIndex, isCjk } from './cin';
import { NgramModel } from './lm';
import { DEFAULT_LATTICE_WEIGHTS, LatticeDecoder, LatticeWeights, Ranked } from './lattice';

export interface SessionUnit {
  ch: string;
  rawCode: string;
  /** > 0 when the user explicitly picked a non-first candidate. */
  selectedIndex: number;
}

export interface PolicyConfig {
  minScoreDelta: number;
  maxChangedAbsolute: number;
  maxChangedFraction: number;
}

export const DEFAULT_POLICY: PolicyConfig = {
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
  /** Best alternative even when below threshold (for the debug line). */
  runnerUp?: { text: string; delta: number };
}

export class CorrectionEngine {
  private decoder: LatticeDecoder;

  constructor(
    public index: CodeIndex,
    public lm: NgramModel,
    public weights: LatticeWeights = DEFAULT_LATTICE_WEIGHTS,
    public policy: PolicyConfig = DEFAULT_POLICY,
  ) {
    this.decoder = new LatticeDecoder(index, lm, weights);
  }

  private maxAllowedChanges(len: number): number {
    if (len === 0) return 0;
    // A misplaced space always changes two characters, so short sentences still allow two.
    const byFrac = Math.floor(len * this.policy.maxChangedFraction);
    return Math.max(Math.min(byFrac, this.policy.maxChangedAbsolute), Math.min(2, len));
  }

  private select(units: SessionUnit[], ranked: Ranked[], t0: number): Suggestion | null {
    const original = units.map((u) => u.ch).join('');
    const originalScore = this.lm.scoreSentence(original);
    const maxChanges = this.maxAllowedChanges(units.length);
    let runnerUp: Suggestion['runnerUp'];
    for (const cand of ranked) {
      if (cand.text === original || cand.changedCount === 0 || cand.changedCount > maxChanges) continue;
      const delta = cand.score - originalScore;
      if (!runnerUp) runnerUp = { text: cand.text, delta };
      if (delta < this.policy.minScoreDelta) continue;
      const changed: number[] = [];
      cand.chars.forEach((c, i) => {
        if (c !== units[i].ch) changed.push(i);
      });
      return { original, corrected: cand.text, changed, score: cand.score, originalScore, changedCount: cand.changedCount, elapsedMs: Date.now() - t0 };
    }
    return runnerUp ? { original, corrected: original, changed: [], score: originalScore, originalScore, changedCount: 0, elapsedMs: Date.now() - t0, runnerUp } : null;
  }

  /** Synchronous decode (used by scripts/tests). Returns a suggestion only when it clears the threshold. */
  suggest(units: SessionUnit[]): Suggestion | null {
    const t0 = Date.now();
    if (units.length === 0) return null;
    if (shouldSkipText(units.map((u) => u.ch).join(''))) return null;
    const s = this.select(units, this.decoder.decode(units), t0);
    return s && s.changed.length ? s : null;
  }

  /**
   * Cooperative decode for the UI: yields between units so typing stays smooth,
   * returns null when cancelled. The result may carry only `runnerUp` when no
   * candidate cleared the threshold.
   */
  async suggestAsync(units: SessionUnit[], isCancelled: () => boolean): Promise<Suggestion | null> {
    const t0 = Date.now();
    if (units.length === 0) return null;
    if (shouldSkipText(units.map((u) => u.ch).join(''))) return null;
    const ranked = await this.decoder.decodeAsync(units, isCancelled);
    if (ranked === null) return null;
    return this.select(units, ranked, t0);
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
