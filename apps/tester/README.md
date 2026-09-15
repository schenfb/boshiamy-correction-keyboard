# In-app keyboard tester (Expo)

A throwaway Expo app for feeling out the Boshiamy-compatible input flow and the
whole-sentence correction bar before the real iOS keyboard extension exists.
It draws its own keyboard inside the app; it is **not** a system keyboard.

The correction engine here is a TypeScript port of `boshiamy_core`
(`src/engine/`). `scripts/check-engine.ts` cross-checks it against the Rust CLI
on identical inputs so the two stay in sync. The product keyboard will call the
Rust core through FFI instead.

## Local-only assets (git-ignored)

```
assets/data/table.cin        your own Boshiamy-style .cin (never committed)
assets/data/zhwiki_tw.bslm   trained by tools/lm (see tools/lm/README.md)
```

## Run

```bash
cd apps/tester
npm install
npx expo start            # scan the QR code with Expo Go on the phone
npx tsx scripts/check-engine.ts   # engine parity check against the Rust CLI
```

## Input behaviour

- Letters (and `,`) build the current code, shown in blue. Space commits the
  first candidate; tapping a candidate commits that one and marks it as an
  explicit selection.
- Every committed character keeps its raw code in the current sentence session.
  `。！？` and Enter end the session; `，、` stay in the session as context.
- The yellow bar shows one whole-sentence correction when the engine's score
  delta clears the threshold; changed characters are red. Tap to replace.
- The grey status line shows session length, correction latency, and the score
  delta so tuning decisions can be made from the phone.
