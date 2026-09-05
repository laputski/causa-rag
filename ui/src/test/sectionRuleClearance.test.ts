import { describe, expect, it } from 'vitest'
import { readFileSync, readdirSync, statSync } from 'node:fs'
import { join } from 'node:path'

/**
 * A section rule sets no bottom margin, so a sibling that brings no clearance
 * of its own sits on the line. `styles.css` lists the siblings that bring none
 * and gives them a margin.
 *
 * What this can check is that the list has not rotted: every selector in it
 * names a class the application still renders. What it cannot check is whether
 * the list is complete, because completeness is a layout question and jsdom
 * lays nothing out. The list was produced by measuring the live document, and
 * reading the markup found six of the eleven entries: the rest are rendered
 * inside conditional branches, so which element follows a rule depends on the
 * data on screen. That procedure is recorded beside the rule in the stylesheet.
 */
const root = join(__dirname, '..')

function sources(dir: string): string[] {
  return readdirSync(dir).flatMap(name => {
    const path = join(dir, name)
    if (statSync(path).isDirectory()) return name === 'test' ? [] : sources(path)
    return /\.tsx?$/.test(name) ? [path] : []
  })
}

const markup = sources(root).map(p => readFileSync(p, 'utf8')).join('\n')
const css = readFileSync(join(root, 'styles.css'), 'utf8')

const listed = [...css.matchAll(/\.section-rule \+ \.([a-z0-9-]+)/g)].map(m => m[1])

describe('clearance under a section rule', () => {
  it('lists at least the siblings measured to need it', () => {
    expect(listed.length).toBeGreaterThanOrEqual(10)
    expect(listed).toContain('chip-row')
  })

  it('names only classes the application still renders', () => {
    const dead = listed.filter(name => !markup.includes(name))
    expect(dead, `selectors for classes nothing renders any more: ${dead.join(', ')}`).toEqual([])
  })

  it('leaves the rule itself without a bottom margin', () => {
    // Giving it one would add to the clearance every other sibling already
    // has, and the padding-based ones cannot be neutralised as cleanly.
    const block = css.match(/\.section-rule \{[^}]*\}/)?.[0] ?? ''
    expect(block).toContain('margin: 20px 0 0')
  })
})
