import React from 'react';
import { Pressable, StyleSheet, Text, View } from 'react-native';

export type KeyAction =
  | { type: 'letter'; value: string }
  | { type: 'space' }
  | { type: 'backspace' }
  | { type: 'enter' }
  | { type: 'punct'; value: string }
  | { type: 'digit'; value: number }
  | { type: 'clear' };

const ROWS = ['qwertyuiop', 'asdfghjkl', 'zxcvbnm'];
// Common Taiwanese punctuation; full-width, committed directly.
const PUNCT = ['，', '。', '、', '？', '！', '：', '；', '「', '」'];

interface Props {
  onKey: (action: KeyAction) => void;
  composing: boolean;
}

function KeyboardImpl({ onKey, composing }: Props) {
  return (
    <View style={styles.keyboard}>
      <View style={styles.row}>
        {PUNCT.map((p) => (
          <Key key={p} label={p} flex={1} onPress={() => onKey({ type: 'punct', value: p })} small />
        ))}
        <Key label="," flex={1} onPress={() => onKey({ type: 'letter', value: ',' })} small />
      </View>
      {ROWS.map((row, i) => (
        <View key={row} style={[styles.row, i === 1 && { paddingHorizontal: 18 }, i === 2 && { paddingHorizontal: 4 }]}>
          {i === 2 && <Key label="清除" flex={1.5} dark onPress={() => onKey({ type: 'clear' })} small />}
          {Array.from(row).map((k) => (
            <Key key={k} label={k} flex={1} onPress={() => onKey({ type: 'letter', value: k })} />
          ))}
          {i === 2 && <Key label="⌫" flex={1.5} dark onPress={() => onKey({ type: 'backspace' })} />}
        </View>
      ))}
      <View style={styles.row}>
        <Key label={composing ? '空白＝送字' : '空白'} flex={5} onPress={() => onKey({ type: 'space' })} />
        <Key label="換行" flex={1.6} dark onPress={() => onKey({ type: 'enter' })} small />
      </View>
    </View>
  );
}

export const Keyboard = React.memo(KeyboardImpl);

const Key = React.memo(function Key({
  label,
  onPress,
  flex,
  dark,
  small,
}: {
  label: string;
  onPress: () => void;
  flex: number;
  dark?: boolean;
  small?: boolean;
}) {
  return (
    <Pressable
      onPress={onPress}
      style={({ pressed }) => [styles.key, { flex }, dark && styles.keyDark, pressed && styles.keyPressed]}
    >
      <Text style={[styles.keyText, small && styles.keyTextSmall]}>{label}</Text>
    </Pressable>
  );
});

const styles = StyleSheet.create({
  keyboard: { backgroundColor: '#d1d4da', paddingVertical: 6, paddingHorizontal: 2, gap: 8 },
  row: { flexDirection: 'row', gap: 5, paddingHorizontal: 2 },
  key: {
    backgroundColor: '#fff',
    borderRadius: 6,
    height: 44,
    alignItems: 'center',
    justifyContent: 'center',
    shadowColor: '#000',
    shadowOpacity: 0.25,
    shadowOffset: { width: 0, height: 1 },
    shadowRadius: 0,
  },
  keyDark: { backgroundColor: '#aeb3bd' },
  keyPressed: { backgroundColor: '#e8e8ea' },
  keyText: { fontSize: 22, color: '#111' },
  keyTextSmall: { fontSize: 15 },
});
