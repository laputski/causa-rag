import { screen, waitFor, fireEvent } from '@testing-library/react'
import { expect } from 'vitest'

// Confirmation stopped being `window.confirm` and became an in-page dialog
// (ui/src/components/ConfirmDialog.tsx). Tests that stubbed the global through
// `vi.stubGlobal('confirm', …)` stopped checking anything after that, and
// stopped silently: stubbing a call that no longer happens does not fail.
//
// One helper for all such tests. There are six, and six copies of the
// button-finding logic would drift apart on the first edit to the dialog's
// markup.

/** Clicks the confirm button in the dialog that opened. */
export async function acceptConfirm() {
  const dialog = await screen.findByRole('alertdialog')
  const button = dialog.querySelector<HTMLButtonElement>('button.btn-danger, button.btn-primary')
  if (!button) throw new Error('the confirm dialog has no confirm button')
  fireEvent.click(button)
  await waitFor(() => expect(screen.queryByRole('alertdialog')).toBeNull())
}

/** Clicks Cancel, the last button in the dialog. */
export async function declineConfirm() {
  const dialog = await screen.findByRole('alertdialog')
  const buttons = [...dialog.querySelectorAll<HTMLButtonElement>('button')]
  fireEvent.click(buttons[buttons.length - 1])
  await waitFor(() => expect(screen.queryByRole('alertdialog')).toBeNull())
}
