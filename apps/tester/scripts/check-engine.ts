// Cross-check the TypeScript engine port against the Rust CLI on the same inputs.
// Usage: npx tsx scripts/check-engine.ts
import { readFileSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { CodeIndex } from '../src/engine/cin';
import { NgramModel } from '../src/engine/lm';
import { CorrectionEngine, SessionUnit } from '../src/engine/engine';

const cinPath = 'assets/data/table.cin';
const lmPath = 'assets/data/zhwiki_tw.bslm';
const t0 = Date.now();
const index = CodeIndex.parse(readFileSync(cinPath, 'utf8'));
const lm = NgramModel.fromBytes(new Uint8Array(readFileSync(lmPath)));
console.log(`loaded table=${index.entries} lm vocab=${lm.vocabSize} bi=${lm.bigramCount} tri=${lm.trigramCount} in ${Date.now() - t0}ms`);
const engine = new CorrectionEngine(index, lm);

const cases = [
  '數:mgp,學:snz,在:xy,許:iwj,多:cca,領:apt,圵:yfe,都:yb,有:x,應:ia,用:nqj',
  '數:mgp,學:snz,有:x,著:rd,善:bv,遠:yw,的:d,歷:ldx,史:cx',
  '數:mgp,學:snz,的:d,重:gq,心:ha,從:mz,求:na,解:nbh,實:nbd,際:bja,蛔:coo,題:dt,轉:cqa,變:lfp',
  '這:xg,樣:dcbr,如:v,果:mr,偶:opv,爾:kad,打:eg,錯:ib,一:e,個:mrq,字:oq,也:na,沒:vfl,關:bdd,係:mm',
];
let ok = 0;
for (const spec of cases) {
  const units: SessionUnit[] = spec.split(',').map((p) => {
    const [ch, rawCode] = p.split(':');
    return { ch, rawCode, selectedIndex: 0 };
  });
  const t1 = Date.now();
  const s = engine.suggest(units);
  const ts = s ? s.corrected : 'none';
  const rust = execFileSync('../../target/release/boshiamy-correct', ['--cin', cinPath, '--lm', lmPath, '--units', spec], { encoding: 'utf8' }).trim();
  const same = ts === rust;
  if (same) ok++;
  console.log(`${same ? 'OK ' : 'DIFF'} ts=${ts} rust=${rust} (${Date.now() - t1}ms${s ? `, delta=${(s.score - s.originalScore).toFixed(2)}` : ''})`);
}
console.log(`${ok}/${cases.length} match`);
