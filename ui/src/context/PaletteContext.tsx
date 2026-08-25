import { createContext, useContext, useState, useEffect, ReactNode } from 'react'

// A palette is the second axis of the look, orthogonal to light and dark. The
// two are separate because they answer different questions: light or dark is
// about the room, a palette is about taste, and collapsing them into one list
// of ten options would make choosing a mood also change the brightness.
//
// The four borrowed palettes are the ones already in use in a sibling
// product's interface, so a person moving between the two keeps their
// setting's meaning. `causa` is this platform's own: a near-neutral ground
// with a single accent, in the manner of a technology registry's scholarly
// styling.
//
// That registry's own note is worth repeating, because it is a warning and not
// a preference: it began with four palettes and came down to one pair, since
// four sets of colours could not be kept consistent. What keeps this from
// repeating here is that every palette redefines exactly the same token names
// and nothing else — no component ever reads a palette by name.
export const PALETTES = ['causa', 'nord', 'everforest', 'gruvbox', 'catppuccin'] as const
export type Palette = (typeof PALETTES)[number]

const STORAGE_KEY = 'rag-platform-palette'
const DEFAULT: Palette = 'causa'

interface PaletteContextValue {
  palette: Palette
  setPalette: (p: Palette) => void
}

function initialPalette(): Palette {
  const stored = localStorage.getItem(STORAGE_KEY)
  return (PALETTES as readonly string[]).includes(stored ?? '') ? (stored as Palette) : DEFAULT
}

const PaletteContext = createContext<PaletteContextValue>({
  palette: DEFAULT,
  setPalette: () => {},
})

export function PaletteProvider({ children }: { children: ReactNode }) {
  const [palette, setPaletteState] = useState<Palette>(initialPalette)

  // The attribute is always present, including for the default: styles.css
  // keys every palette off `[data-palette="…"]`, and leaving the default
  // implicit would mean one palette's tokens lived in a different place from
  // the other four's.
  useEffect(() => {
    document.documentElement.setAttribute('data-palette', palette)
    localStorage.setItem(STORAGE_KEY, palette)
  }, [palette])

  return (
    <PaletteContext.Provider value={{ palette, setPalette: setPaletteState }}>
      {children}
    </PaletteContext.Provider>
  )
}

export function usePalette() {
  return useContext(PaletteContext)
}
