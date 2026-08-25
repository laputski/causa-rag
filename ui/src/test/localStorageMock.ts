// jsdom does not expose a global `localStorage` in this vitest setup; provide a
// minimal in-memory implementation so components that persist UI preferences
// (theme, language) can be unit-tested. Split into its own module (rather than
// inlined in setup.ts) so it runs before `../i18n`'s side-effecting `init()`
// call, which reads localStorage — sibling imports in the same file execute
// in write order, but only relative to each other, before any inline code.
if (typeof globalThis.localStorage === 'undefined') {
  const store = new Map<string, string>()
  const mock: Storage = {
    getItem: (k) => (store.has(k) ? store.get(k)! : null),
    setItem: (k, v) => { store.set(k, String(v)) },
    removeItem: (k) => { store.delete(k) },
    clear: () => { store.clear() },
    key: (i) => Array.from(store.keys())[i] ?? null,
    get length() { return store.size },
  }
  Object.defineProperty(globalThis, 'localStorage', { value: mock, configurable: true })
}
