// Loader for the BSLM v1 character trigram format written by boshiamy_lm_train.
// Mirrors crates/boshiamy_core/src/ngram_model.rs. 64-bit keys are kept as
// (hi, lo) u32 pairs so no BigInt is needed on Hermes.

export const BOS = 0;
export const EOS = 1;
export const UNK = 2;
const FIRST_CHAR_ID = 3;

export class NgramModel {
  private charToId = new Map<string, number>();
  private unkLogprob = 0;
  private uniLp!: Float32Array;
  private uniBo!: Float32Array;
  private biHi!: Uint32Array;
  private biLo!: Uint32Array;
  private biLp!: Float32Array;
  private biBo!: Float32Array;
  private triHi!: Uint32Array;
  private triLo!: Uint32Array;
  private triLp!: Float32Array;
  vocabSize = 0;
  bigramCount = 0;
  trigramCount = 0;

  static fromBytes(bytes: Uint8Array): NgramModel {
    const m = new NgramModel();
    const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    let pos = 0;
    const magic = String.fromCharCode(bytes[0], bytes[1], bytes[2], bytes[3]);
    if (magic !== 'BSLM') throw new Error('not a BSLM model');
    pos = 4;
    const u32 = () => {
      const v = dv.getUint32(pos, true);
      pos += 4;
      return v;
    };
    const f32 = () => {
      const v = dv.getFloat32(pos, true);
      pos += 4;
      return v;
    };
    const f32s = (n: number) => {
      const out = new Float32Array(n);
      for (let i = 0; i < n; i++) out[i] = dv.getFloat32(pos + i * 4, true);
      pos += n * 4;
      return out;
    };
    const u64s = (n: number): [Uint32Array, Uint32Array] => {
      const hi = new Uint32Array(n);
      const lo = new Uint32Array(n);
      for (let i = 0; i < n; i++) {
        lo[i] = dv.getUint32(pos + i * 8, true);
        hi[i] = dv.getUint32(pos + i * 8 + 4, true);
      }
      pos += n * 8;
      return [hi, lo];
    };
    const version = u32();
    if (version !== 1) throw new Error(`unsupported BSLM version ${version}`);
    const order = u32();
    if (order !== 3) throw new Error(`unsupported order ${order}`);
    const nVocab = u32();
    for (let i = 0; i < nVocab; i++) {
      const cp = u32();
      m.charToId.set(String.fromCodePoint(cp), FIRST_CHAR_ID + i);
    }
    m.vocabSize = nVocab;
    m.unkLogprob = f32();
    const total = nVocab + FIRST_CHAR_ID;
    m.uniLp = f32s(total);
    m.uniBo = f32s(total);
    const nBi = u32();
    [m.biHi, m.biLo] = u64s(nBi);
    m.biLp = f32s(nBi);
    m.biBo = f32s(nBi);
    m.bigramCount = nBi;
    const nTri = u32();
    [m.triHi, m.triLo] = u64s(nTri);
    m.triLp = f32s(nTri);
    m.trigramCount = nTri;
    return m;
  }

  id(ch: string): number {
    return this.charToId.get(ch) ?? UNK;
  }

  private find(hi: Uint32Array, lo: Uint32Array, kh: number, kl: number): number {
    let l = 0;
    let r = hi.length - 1;
    while (l <= r) {
      const mid = (l + r) >>> 1;
      const h = hi[mid];
      if (h < kh || (h === kh && lo[mid] < kl)) l = mid + 1;
      else if (h === kh && lo[mid] === kl) return mid;
      else r = mid - 1;
    }
    return -1;
  }

  private bi(a: number, b: number): number {
    return this.find(this.biHi, this.biLo, a, b >>> 0);
  }

  private tri(a: number, b: number, c: number): number {
    // key = a<<40 | b<<20 | c  →  hi = (a<<8) | (b>>>12), lo = ((b & 0xfff)<<20) | c
    const hi = ((a << 8) | (b >>> 12)) >>> 0;
    const lo = (((b & 0xfff) << 20) | c) >>> 0;
    return this.find(this.triHi, this.triLo, hi, lo);
  }

  /** log P(c | a b) with backoff (natural log). */
  logprob(a: number, b: number, c: number): number {
    const t = this.tri(a, b, c);
    if (t >= 0) return this.triLp[t];
    const ab = this.bi(a, b);
    const boAb = ab >= 0 ? this.biBo[ab] : 0;
    const bc = this.bi(b, c);
    if (bc >= 0) return boAb + this.biLp[bc];
    const uni = c === UNK ? this.unkLogprob : this.uniLp[c];
    return boAb + this.uniBo[b] + uni;
  }

  scoreIds(ids: number[]): number {
    let a = BOS;
    let b = BOS;
    let total = 0;
    for (const c of ids) {
      total += this.logprob(a, b, c);
      a = b;
      b = c;
    }
    return total + this.logprob(a, b, EOS);
  }

  scoreSentence(text: string): number {
    return this.scoreIds(Array.from(text).map((c) => this.id(c)));
  }

  /** Unigram prior used to rank capped candidates. */
  unigram(ch: string): number {
    const c = this.id(ch);
    return c === UNK ? this.unkLogprob : this.uniLp[c];
  }
}
