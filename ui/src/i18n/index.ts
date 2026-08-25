import i18n from 'i18next'
import { initReactI18next } from 'react-i18next'

import en from './locales/en.json'
import ru from './locales/ru.json'

// `ru` is the source language and carries every key; `en` is a full
// translation of it.
//
// Five more files exist — be/de/es/fr/zh — and each covers 176 keys of 2012,
// which is nine per cent: the sidebar and little else. They were offered in
// the switcher, so picking 中文 gave a Chinese menu and an English everything,
// which reads as broken rather than as untranslated. A language belongs in
// this list when it is translated, not when its file exists; the files stay
// as the head start they are. Held by i18nCoverage.test.ts.
export const SUPPORTED_LANGUAGES = [
  { code: 'ru', label: 'Русский' },
  { code: 'en', label: 'English' },
] as const

const STORAGE_KEY = 'rag-platform-lang'

function initialLanguage(): string {
  const stored = localStorage.getItem(STORAGE_KEY)
  if (stored && SUPPORTED_LANGUAGES.some(l => l.code === stored)) return stored
  // English, matching `fallbackLng` below and the rest of the repository. A
  // stored choice always wins, so anyone who has already picked a language
  // keeps it; this only decides what a fresh install opens in.
  return 'en'
}

i18n.use(initReactI18next).init({
  resources: {
    en: { translation: en },
    ru: { translation: ru },
  },
  lng: initialLanguage(),
  fallbackLng: 'en',
  interpolation: { escapeValue: false },
  returnEmptyString: false,
})

i18n.on('languageChanged', lng => {
  localStorage.setItem(STORAGE_KEY, lng)
})

export default i18n
