import { Routes, Route } from 'react-router-dom'
import Layout from '@/components/Layout'
import HomePage from '@/pages/HomePage'
import ArtistiPage from '@/pages/ArtistiPage'
import StoricoPage from '@/pages/StoricoPage'
import TeamBuilderPage from '@/pages/TeamBuilderPage'

function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/artisti" element={<ArtistiPage />} />
        <Route path="/storico/:artistaId" element={<StoricoPage />} />
        <Route path="/team-builder" element={<TeamBuilderPage />} />
      </Routes>
    </Layout>
  )
}

export default App
