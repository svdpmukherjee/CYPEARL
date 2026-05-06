export function isEmailFilled(e) {
  return !!(e.subject?.trim() && e.content?.trim() && e.frequency);
}

export function isSenderComplete(sender) {
  return sender.role.trim() && sender.type && sender.emails.some(isEmailFilled);
}

export function formatBonus(pence) {
  if (pence <= 0) return "£0.00";
  return `£${(pence / 100).toFixed(2)}`;
}
