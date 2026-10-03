import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import ResourceForm, { STORED_SECRET, type FieldSchema, type ResourceValues } from '../components/ResourceForm'

// The form of one Realm resource.
//
// The subject is what a save sends for a secret. The server reads the mask as
// "keep what is stored" and an empty string as "remove it", so a form that
// sent an empty string for a field nobody touched would erase the key on
// every save of the host.

const SCHEMA: Record<string, FieldSchema> = {
  base_url: { type: 'string' },
  api_key: { type: 'string', secret: true },
  https: { type: 'boolean', default: false },
  port: { type: 'integer', default: 6333 },
}

const STORED: ResourceValues = { type: 'x', base_url: 'https://models.example/v1', api_key: STORED_SECRET, https: true, port: 443 }

function show(initial?: ResourceValues) {
  const onSave = vi.fn()
  render(<ResourceForm schema={SCHEMA} initial={initial} saving={false} onSave={onSave} onCancel={() => {}} />)
  const saved = () => { fireEvent.click(screen.getByText('Save')); return onSave.mock.calls[onSave.mock.calls.length - 1][0] as ResourceValues }
  return { saved }
}

describe('ResourceForm', () => {
  it('keeps a stored secret when nobody touches it', () => {
    const { saved } = show(STORED)
    expect(screen.getByText('stored')).toBeInTheDocument()
    fireEvent.change(screen.getByLabelText('Address'), { target: { value: 'https://other.example/v1' } })
    expect(saved()).toMatchObject({ base_url: 'https://other.example/v1', api_key: STORED_SECRET })
  })

  it('never shows the stored secret, not even as dots in an input', () => {
    show(STORED)
    expect(document.querySelector('input[type="password"]')).toBeNull()
    expect(screen.queryByDisplayValue(STORED_SECRET)).toBeNull()
  })

  it('replaces a stored secret with what is typed', () => {
    const { saved } = show(STORED)
    fireEvent.click(screen.getByText('Replace'))
    fireEvent.change(screen.getByLabelText('Key'), { target: { value: 'new-key' } })
    expect(saved().api_key).toBe('new-key')
  })

  it('keeps the stored secret when Replace is opened and nothing is typed', () => {
    const { saved } = show(STORED)
    fireEvent.click(screen.getByText('Replace'))
    expect(saved().api_key).toBe(STORED_SECRET)
    fireEvent.change(screen.getByLabelText('Key'), { target: { value: 'x' } })
    fireEvent.change(screen.getByLabelText('Key'), { target: { value: '' } })
    expect(saved().api_key).toBe(STORED_SECRET)
  })

  it('removes a stored secret only through its own button, and can take that back', () => {
    const { saved } = show(STORED)
    fireEvent.click(screen.getByText('Remove'))
    expect(screen.getByText('will be removed on save')).toBeInTheDocument()
    expect(saved().api_key).toBe('')
    fireEvent.click(screen.getByText('Keep the stored one'))
    expect(saved().api_key).toBe(STORED_SECRET)
  })

  it('takes a new secret in a password field when none is stored', () => {
    const { saved } = show()
    const key = screen.getByLabelText('Key') as HTMLInputElement
    expect(key.type).toBe('password')
    expect(saved().api_key).toBe('')
    fireEvent.change(key, { target: { value: 'first-key' } })
    expect(saved().api_key).toBe('first-key')
  })

  it('sends a yes-or-no field as a boolean and starts it from what is stored', () => {
    const { saved } = show(STORED)
    const https = screen.getByLabelText('Connect over HTTPS') as HTMLInputElement
    expect(https.type).toBe('checkbox')
    expect(https.checked).toBe(true)
    fireEvent.click(https)
    expect(saved().https).toBe(false)
  })

  it('starts a new resource from the defaults of its type', () => {
    const { saved } = show()
    expect(saved()).toMatchObject({ port: '6333', https: false, base_url: '' })
  })

  it('names a field in words and shows an unnamed one by its own name', () => {
    render(<ResourceForm schema={{ shard_hint: { type: 'string' } }} saving={false} onSave={() => {}} onCancel={() => {}} />)
    expect(screen.getByLabelText('shard_hint')).toBeInTheDocument()
  })
})
