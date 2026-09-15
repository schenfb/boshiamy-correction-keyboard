// Minimal .cin parser + bidirectional index with a wildcard neighbor index.
// Mirrors crates/boshiamy_core (cin_parser.rs, code_index.rs).

export class CodeIndex {
  private codeToChars = new Map<string, string[]>();
  private charToCodes = new Map<string, string[]>();
  private wildcard = new Map<string, Array<[string, string]>>();
  name = '';
  entries = 0;

  static parse(text: string): CodeIndex {
    const index = new CodeIndex();
    let inChardef = false;
    for (const raw of text.split(/\r?\n/)) {
      const line = raw.trim();
      if (!line || line.startsWith('#')) continue;
      const lower = line.toLowerCase();
      if (lower === '%chardef begin') {
        inChardef = true;
        continue;
      }
      if (lower === '%chardef end') {
        inChardef = false;
        continue;
      }
      if (!inChardef) {
        if (lower.startsWith('%cname ')) index.name = line.slice(7).trim();
        continue;
      }
      const parts = line.split(/\s+/);
      if (parts.length < 2) continue;
      const code = parts[0].toLowerCase();
      const chars = Array.from(parts[1]);
      if (chars.length !== 1) continue;
      index.insert(code, chars[0]);
    }
    return index;
  }

  insert(code: string, ch: string) {
    push(this.codeToChars, code, ch);
    push(this.charToCodes, ch, code);
    for (const pat of wildcardPatterns(code)) {
      const list = this.wildcard.get(pat);
      if (list) list.push([code, ch]);
      else this.wildcard.set(pat, [[code, ch]]);
    }
    this.entries++;
  }

  charsForCode(code: string): string[] {
    return this.codeToChars.get(code) ?? [];
  }

  codesForChar(ch: string): string[] {
    return this.charToCodes.get(ch) ?? [];
  }

  /** Any code that has at least one char starting with `prefix` (for a live candidate preview). */
  hasPrefix(prefix: string): boolean {
    for (const code of this.codeToChars.keys()) if (code.startsWith(prefix)) return true;
    return false;
  }

  /** (code, char) pairs exactly one substitution away from rawCode; exact matches excluded. */
  substitutionNeighbors(rawCode: string): Array<[string, string]> {
    const out: Array<[string, string]> = [];
    for (const pat of wildcardPatterns(rawCode)) {
      const list = this.wildcard.get(pat);
      if (!list) continue;
      for (const pair of list) if (pair[0] !== rawCode) out.push(pair);
    }
    return out;
  }
}

function push(map: Map<string, string[]>, key: string, value: string) {
  const list = map.get(key);
  if (list) {
    if (!list.includes(value)) list.push(value);
  } else map.set(key, [value]);
}

function wildcardPatterns(code: string): string[] {
  const out: string[] = [];
  for (let i = 0; i < code.length; i++) out.push(code.slice(0, i) + '?' + code.slice(i + 1));
  return out;
}

export function isCjk(ch: string): boolean {
  const u = ch.codePointAt(0) ?? 0;
  return (u >= 0x4e00 && u <= 0x9fff) || (u >= 0x3400 && u <= 0x4dbf);
}
