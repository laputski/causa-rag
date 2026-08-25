import { describe, it, expect } from 'vitest'
import ru from '../i18n/locales/ru.json'
import en from '../i18n/locales/en.json'

type Tree = { [key: string]: Tree | string }

function leafPaths(tree: Tree, prefix = ''): string[] {
  return Object.entries(tree).flatMap(([k, v]) => {
    const full = prefix ? `${prefix}.${k}` : k
    return typeof v === 'string' ? [full] : leafPaths(v as Tree, full)
  })
}

// Vite-native file scan (no Node `fs`/`path` typings in this browser-only
// project) — grabs raw source text for every non-test, non-i18n source file.
const SOURCE_FILES = import.meta.glob('../**/*.{ts,tsx}', {
  query: '?raw',
  import: 'default',
  eager: true,
}) as Record<string, string>

const NAMESPACES = Object.keys(ru)
const KEY_LITERAL_RE = new RegExp(
  `['"](${NAMESPACES.join('|')})((?:\\.[a-zA-Z][a-zA-Z0-9]*)+)['"]`,
  'g',
)

function referencedKeys(): Set<string> {
  const found = new Set<string>()
  for (const [filePath, content] of Object.entries(SOURCE_FILES)) {
    if (filePath.includes('/test/') || filePath.includes('/i18n/')) continue
    for (const m of content.matchAll(KEY_LITERAL_RE)) {
      found.add(m[1] + m[2])
    }
  }
  return found
}

function resolves(tree: Tree, dotted: string): boolean {
  const parts = dotted.split('.')
  let node: Tree | string = tree
  for (let i = 0; i < parts.length; i++) {
    if (typeof node !== 'object' || node === null || !(parts[i] in node)) {
      // pluralized leaf: base key itself is absent, but `<base>_one` etc. is
      if (typeof node === 'object' && node !== null && i === parts.length - 1) {
        return Object.keys(node).some((k) => k.startsWith(parts[i] + '_'))
      }
      return false
    }
    node = node[parts[i]]
  }
  return typeof node === 'string'
}

describe('i18n coverage', () => {
  it('ru.json and en.json declare exactly the same set of keys', () => {
    // Plural forms are compared by their stem: Russian has three
    // (`one`/`few`/`many`) and English two (`one`/`other`), so demanding the
    // suffixes match would demand Russian grammar of English, which is exactly
    // the mistake that had been made (see the next assertion).
    const base = (k: string) => k.replace(/_(one|two|few|many|other|zero)$/, '')
    const ruKeys = new Set(leafPaths(ru as Tree).map(base))
    const enKeys = new Set(leafPaths(en as Tree).map(base))
    const missingInEn = [...ruKeys].filter((k) => !enKeys.has(k)).sort()
    const missingInRu = [...enKeys].filter((k) => !ruKeys.has(k)).sort()
    expect(missingInEn).toEqual([])
    expect(missingInRu).toEqual([])
  })

  it('the switcher lists only languages that are translated through', async () => {
    // Five files (be/de/es/fr/zh) cover 176 keys of 2012, which is nine per
    // cent: the sidebar and little else. They used to stand in the switcher, so
    // picking 中文 gave a Chinese menu and an English everything, which reads as
    // broken, and not as untranslated.
    const { SUPPORTED_LANGUAGES } = await import('../i18n')
    const base = leafPaths(ru as Tree).length
    for (const { code } of SUPPORTED_LANGUAGES) {
      const tree = (await import(`../i18n/locales/${code}.json`)).default as Tree
      const share = leafPaths(tree).length / base
      expect(share, `${code}: ${Math.round(share * 100)} % translated`).toBeGreaterThan(0.95)
    }
  })

  it('the English file carries no untranslated Russian', () => {
    // Found live while counting the guide's sections: two of them, `realm` and
    // `shell`, sat in en.json entirely in Russian — twenty values. The key-parity
    // check did not see them, because the keys were all present. And the five
    // languages with no file of their own fall back to English, so they were
    // showing the Russian too.
    const cyrillic = /[А-Яа-яЁё]/
    const offenders = leafPaths(en as Tree).filter(path => {
      const value = path.split('.').reduce<unknown>((node, key) => (node as Tree)?.[key], en)
      return typeof value === 'string' && cyrillic.test(value)
    })
    expect(offenders).toEqual([])
  })

  it("every plural stem carries its own language's forms", () => {
    // Found live: `datasetsPage.questionsCount` was defined as
    // `_one`/`_few`/`_many` in both files. English selects none of
    // the three, so i18next printed the raw key: "v0 datasetsPage.questionsCount
    // full" appeared in the dataset list and in the page caption.
    const REQUIRED: Record<string, string[]> = { ru: ['one', 'few', 'many'], en: ['one', 'other'] }
    const groups = (tree: Tree) => {
      const out = new Map<string, Set<string>>()
      for (const path of leafPaths(tree)) {
        const m = path.match(/^(.*)_(one|two|few|many|other|zero)$/)
        if (!m) continue
        if (!out.has(m[1])) out.set(m[1], new Set())
        out.get(m[1])!.add(m[2])
      }
      return out
    }
    const missing: string[] = []
    for (const [lang, tree] of [['ru', ru], ['en', en]] as [string, Tree][]) {
      for (const [base, forms] of groups(tree)) {
        for (const form of REQUIRED[lang]) {
          if (!forms.has(form)) missing.push(`${lang}: ${base} has no _${form}`)
        }
      }
    }
    expect(missing).toEqual([])
  })

  it('has no empty-string translation values', () => {
    const empties = (tree: Tree, prefix = ''): string[] =>
      Object.entries(tree).flatMap(([k, v]) => {
        const full = prefix ? `${prefix}.${k}` : k
        if (typeof v === 'string') return v.trim() === '' ? [full] : []
        return empties(v as Tree, full)
      })
    expect(empties(ru as Tree)).toEqual([])
    expect(empties(en as Tree)).toEqual([])
  })

  it('every i18n-key-shaped string literal in ui/src resolves in both locales', () => {
    const keys = [...referencedKeys()].sort()
    expect(keys.length).toBeGreaterThan(100) // sanity check the scan itself is working
    const missingInRu = keys.filter((k) => !resolves(ru as Tree, k))
    const missingInEn = keys.filter((k) => !resolves(en as Tree, k))
    expect(missingInRu).toEqual([])
    expect(missingInEn).toEqual([])
  })
})
