/**
 * Estrae le iniziali da un nome artista.
 *
 * Regole:
 * - "Fedez" → "FE"
 * - "Fedez & Marco Masini" → "FM"
 * - "J-Ax" → "JA" (trattato come singolo token)
 * - "Achille Lauro" → "AL"
 * - "Angelina Mango" → "AM"
 * - "Dargen D'Amico" → "DD"
 * - "Giorgia" → "GI"
 * - "Mr. Rain" → "MR"
 *
 * @param name - Il nome dell'artista
 * @returns Le iniziali in maiuscolo (massimo 2 caratteri)
 */
export function getInitials(name: string): string {
  if (!name || typeof name !== 'string') {
    return '??'
  }

  // Rimuovi spazi e normalizza
  const trimmed = name.trim()

  if (!trimmed) {
    return '??'
  }

  // Split su spazi, &, e altri separatori comuni
  const tokens = trimmed
    .split(/[\s&+–,/]+/)
    .filter(t => t.length > 0)

  if (tokens.length === 0) {
    return '??'
  }

  // Se c'è un solo token, prendi le prime 2 lettere
  if (tokens.length === 1) {
    const token = tokens[0].toUpperCase()
    // Gestione casi speciali come "J-Ax" → "JA"
    const cleanToken = token.replace(/[^A-Z]/g, '')
    return cleanToken.length >= 2
      ? cleanToken.slice(0, 2)
      : token.padEnd(2, '?').slice(0, 2)
  }

  // Se ci sono più token, prendi la prima lettera dei primi 2
  const first = tokens[0].toUpperCase().charAt(0)
  const second = tokens[1].toUpperCase().charAt(0)

  // Assicurati che entrambe siano lettere valide
  const isValidFirst = /[A-Z]/.test(first)
  const isValidSecond = /[A-Z]/.test(second)

  if (!isValidFirst && !isValidSecond) {
    return '??'
  }

  return (
    (isValidFirst ? first : '?') +
    (isValidSecond ? second : '?')
  )
}
