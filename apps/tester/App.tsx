import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { ActivityIndicator, Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import { SafeAreaProvider, SafeAreaView } from 'react-native-safe-area-context';
import { StatusBar } from 'expo-status-bar';
import { Asset } from 'expo-asset';
import { File } from 'expo-file-system';
import { CodeIndex } from './src/engine/cin';
import { NgramModel } from './src/engine/lm';
import { CorrectionEngine, SessionUnit, Suggestion } from './src/engine/engine';
import { Keyboard, KeyAction } from './src/Keyboard';

const MAX_CODE_LEN = 5;
const SENTENCE_END = new Set(['。', '！', '？']);

async function loadAssetBytes(module: number): Promise<Uint8Array> {
  const asset = await Asset.fromModule(module).downloadAsync();
  if (!asset.localUri) throw new Error('asset has no local uri');
  return new Uint8Array(await new File(asset.localUri).arrayBuffer());
}

async function loadAssetText(module: number): Promise<string> {
  const asset = await Asset.fromModule(module).downloadAsync();
  if (!asset.localUri) throw new Error('asset has no local uri');
  return await new File(asset.localUri).text();
}

export default function App() {
  const [engine, setEngine] = useState<CorrectionEngine | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [loadInfo, setLoadInfo] = useState('載入中…');

  // Committed text, current composing code, and the session units (chars with raw codes).
  const [text, setText] = useState('');
  const [code, setCode] = useState('');
  const unitsRef = useRef<SessionUnit[]>([]);
  const [suggestion, setSuggestion] = useState<Suggestion | null>(null);
  const [status, setStatus] = useState('');
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    (async () => {
      try {
        const t0 = Date.now();
        // Both files are git-ignored local assets; see apps/tester/README.md.
        const cinText = await loadAssetText(require('./assets/data/table.cin'));
        const index = CodeIndex.parse(cinText);
        const lmBytes = await loadAssetBytes(require('./assets/data/zhwiki_tw.bslm'));
        const lm = NgramModel.fromBytes(lmBytes);
        setEngine(new CorrectionEngine(index, lm));
        setLoadInfo(`${index.name || '碼表'} ${index.entries} 條 · LM ${(lmBytes.length / 1e6).toFixed(0)} MB · ${Date.now() - t0} ms`);
      } catch (e) {
        setLoadError(String(e));
      }
    })();
  }, []);

  const candidates = useMemo(() => (engine && code ? engine.index.charsForCode(code) : []), [engine, code]);

  const scheduleSuggest = useCallback(() => {
    if (timer.current) clearTimeout(timer.current);
    timer.current = setTimeout(() => {
      if (!engine) return;
      const units = unitsRef.current;
      const t0 = Date.now();
      const s = engine.suggest(units);
      setSuggestion(s);
      setStatus(`${units.length} 字 · 校正 ${Date.now() - t0} ms${s ? ` · Δ${(s.score - s.originalScore).toFixed(1)}` : ' · 無建議'}`);
    }, 120);
  }, [engine]);

  const commit = useCallback(
    (ch: string, rawCode: string, selectedIndex: number) => {
      setText((t) => t + ch);
      unitsRef.current = [...unitsRef.current, { ch, rawCode, selectedIndex }];
      setCode('');
      scheduleSuggest();
    },
    [scheduleSuggest],
  );

  const endSession = useCallback(() => {
    unitsRef.current = [];
    setSuggestion(null);
    setStatus('');
  }, []);

  const onKey = useCallback(
    (action: KeyAction) => {
      switch (action.type) {
        case 'letter':
          if (code.length < MAX_CODE_LEN) setCode(code + action.value);
          return;
        case 'space':
          if (code) {
            if (candidates.length > 0) commit(candidates[0], code, 0);
            else setCode('');
          } else {
            setText((t) => t + ' ');
            endSession();
          }
          return;
        case 'backspace':
          if (code) {
            setCode(code.slice(0, -1));
            return;
          }
          if (!text) return;
          {
            const chars = Array.from(text);
            const last = chars[chars.length - 1];
            setText(chars.slice(0, -1).join(''));
            const units = unitsRef.current;
            if (units.length && units[units.length - 1].ch === last) {
              unitsRef.current = units.slice(0, -1);
              scheduleSuggest();
            } else {
              endSession();
            }
          }
          return;
        case 'enter':
          setCode('');
          setText((t) => t + '\n');
          endSession();
          return;
        case 'punct':
          setCode('');
          setText((t) => t + action.value);
          if (SENTENCE_END.has(action.value)) endSession();
          else {
            unitsRef.current = [...unitsRef.current, { ch: action.value, rawCode: '', selectedIndex: 0 }];
            scheduleSuggest();
          }
          return;
        case 'clear':
          setCode('');
          setText('');
          endSession();
          return;
        case 'digit':
          return;
      }
    },
    [code, candidates, commit, endSession, scheduleSuggest, text],
  );

  const applySuggestion = useCallback(() => {
    if (!suggestion || !engine) return;
    if (!text.endsWith(suggestion.original)) {
      setStatus('文字已變動，取消替換');
      setSuggestion(null);
      return;
    }
    const corrected = Array.from(suggestion.corrected);
    setText(text.slice(0, text.length - suggestion.original.length) + suggestion.corrected);
    // Keep context for later corrections; the corrected chars now count as "typed".
    unitsRef.current = unitsRef.current.map((u, i) => {
      const ch = corrected[i];
      if (ch === u.ch) return u;
      return { ch, rawCode: engine.index.codesForChar(ch)[0] ?? '', selectedIndex: 0 };
    });
    setSuggestion(null);
    setStatus(`已替換 ${suggestion.changedCount} 字`);
  }, [suggestion, engine, text]);

  return (
    <SafeAreaProvider>
      <SafeAreaView style={styles.root} edges={['top', 'bottom']}>
        <StatusBar style="dark" />
        <View style={styles.header}>
          <Text style={styles.title}>嘸蝦米整句校正 · 測試</Text>
          <Text style={styles.info}>{loadError ?? loadInfo}</Text>
        </View>
        <ScrollView style={styles.editor} contentContainerStyle={{ padding: 16 }}>
          <Text style={styles.text}>
            {text}
            {code ? <Text style={styles.composing}>{code}</Text> : null}
            <Text style={styles.cursor}>▏</Text>
          </Text>
        </ScrollView>
        <Text style={styles.status}>{status}</Text>
        <View style={styles.suggestionBar}>
          {suggestion ? (
            <Pressable onPress={applySuggestion} style={styles.suggestion}>
              <Text style={styles.suggestionLabel}>整句校正 ›</Text>
              <Text style={styles.suggestionText} numberOfLines={2}>
                {Array.from(suggestion.corrected).map((c, i) => (
                  <Text key={i} style={suggestion.changed.includes(i) ? styles.changedChar : undefined}>
                    {c}
                  </Text>
                ))}
              </Text>
            </Pressable>
          ) : (
            <Text style={styles.suggestionEmpty}>{engine ? '' : ''}</Text>
          )}
        </View>
        <ScrollView horizontal style={styles.candidates} contentContainerStyle={styles.candidatesContent} keyboardShouldPersistTaps="always">
          {code && candidates.length === 0 ? <Text style={styles.noCandidate}>沒有「{code}」這個碼</Text> : null}
          {candidates.map((c, i) => (
            <Pressable key={c + i} onPress={() => commit(c, code, i)} style={styles.candidate}>
              <Text style={styles.candidateIndex}>{i + 1}</Text>
              <Text style={styles.candidateChar}>{c}</Text>
            </Pressable>
          ))}
        </ScrollView>
        {engine ? (
          <Keyboard onKey={onKey} composing={!!code} />
        ) : (
          <View style={styles.loading}>{loadError ? null : <ActivityIndicator />}</View>
        )}
      </SafeAreaView>
    </SafeAreaProvider>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: '#f2f2f7' },
  header: { paddingHorizontal: 16, paddingTop: 6, paddingBottom: 4 },
  title: { fontSize: 16, fontWeight: '600' },
  info: { fontSize: 11, color: '#666', marginTop: 2 },
  editor: { flex: 1, backgroundColor: '#fff', marginHorizontal: 12, borderRadius: 10 },
  text: { fontSize: 22, lineHeight: 34, color: '#111' },
  composing: { color: '#0a60ff', backgroundColor: '#dfe9ff' },
  cursor: { color: '#0a60ff' },
  status: { fontSize: 11, color: '#888', paddingHorizontal: 16, paddingTop: 4, height: 20 },
  suggestionBar: { minHeight: 56, paddingHorizontal: 12, paddingVertical: 4 },
  suggestion: { backgroundColor: '#fff7d6', borderRadius: 10, padding: 10, borderWidth: 1, borderColor: '#f0d98a' },
  suggestionLabel: { fontSize: 11, color: '#8a6d00', marginBottom: 2 },
  suggestionText: { fontSize: 19, color: '#111' },
  suggestionEmpty: { fontSize: 12, color: '#aaa', padding: 10 },
  changedChar: { color: '#c1121f', fontWeight: '700', textDecorationLine: 'underline' },
  candidates: { maxHeight: 48, backgroundColor: '#e4e6eb' },
  candidatesContent: { alignItems: 'center', paddingHorizontal: 6 },
  noCandidate: { color: '#999', fontSize: 14, paddingHorizontal: 8 },
  candidate: { flexDirection: 'row', alignItems: 'baseline', paddingHorizontal: 10, paddingVertical: 8 },
  candidateIndex: { fontSize: 10, color: '#999', marginRight: 2 },
  candidateChar: { fontSize: 24, color: '#111' },
  loading: { height: 260, alignItems: 'center', justifyContent: 'center', backgroundColor: '#d1d4da' },
});
