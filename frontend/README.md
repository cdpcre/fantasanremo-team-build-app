# Frontend - FantaSanremo Team Builder

Frontend React/Vite del progetto.

## Scripts

```bash
npm run dev               # sviluppo locale
npm run build             # build produzione (API remota)
npm run build:standalone  # build con dati locali statici (Vercel standalone)
npm run test:run          # test
npm run lint              # lint
```

## Variabili Ambiente

- `VITE_API_URL`: base URL backend (es. `https://api.example.com`)
- `VITE_API_MODE`:
  - `remote` (default): usa API backend
  - `local`: usa `public/data/vercel_snapshot.json`

## Deploy Vercel (frontend classico)

1. Imposta root directory `frontend`
2. Build command `npm run build`
3. Output directory `dist`
4. Aggiungi env `VITE_API_URL`

`vercel.json` include rewrite SPA per supportare route React (`/artisti`, `/team-builder`, ...).
