import { Link } from 'react-router-dom'
import {
  Mic,
  Coins,
  Users,
  Calendar,
  Trophy,
  Sparkles,
  TrendingUp,
  ArrowRight,
  Target,
  Zap,
  Crown
} from 'lucide-react'

export default function HomePage() {
  return (
    <div className="space-y-16 overflow-x-hidden">
      {/* Hero Section */}
      <section className="relative">
        {/* Background gradient */}
        <div className="absolute inset-0 bg-gradient-to-br from-amber-400/5 via-transparent to-navy-800/50 rounded-3xl -z-10" />
        <div className="absolute inset-0 bg-gradient-to-t from-navy-950 via-transparent to-transparent rounded-3xl -z-10" />

        {/* Animated spotlight */}
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-96 h-96 bg-amber-400/10 blur-3xl rounded-full animate-pulse" />

        <div className="text-center py-16">
          {/* Badge */}
          <div className="inline-flex items-center space-x-2 bg-amber-500/10 border border-amber-500/30 rounded-full px-4 py-2 mb-8 animate-fade-in">
            <Sparkles className="w-4 h-4 text-amber-600 dark:text-amber-400" />
            <span className="text-sm font-medium text-amber-600 dark:text-amber-400">Edizione 2026</span>
          </div>

          {/* Main heading */}
          <h2 className="text-5xl md:text-6xl lg:text-7xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-amber-600 via-amber-500 to-amber-600 dark:from-amber-400 dark:via-amber-500 dark:to-amber-400 mb-6 animate-slide-up">
            FantaSanremo
            <br />
            Team Builder 2026!
          </h2>

          <p className="text-xl md:text-2xl text-gray-800 dark:text-gray-200 max-w-3xl mx-auto mb-8 leading-relaxed animate-slide-up animation-delay-200">
            Costruisci la tua squadra ideale con i 30 Big artisti in gara.
            <br />
            <span className="text-amber-600 dark:text-amber-400">Analizza lo storico</span>, consulta le{' '}
            <span className="text-amber-600 dark:text-amber-400">predizioni ML</span> e vinci il tuo torneo!
          </p>

          {/* CTA Buttons */}
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4 animate-slide-up animation-delay-400">
            <Link
              to="/team-builder"
              className="group relative inline-flex items-center space-x-2 bg-gradient-to-r from-amber-500 to-amber-600 hover:from-amber-400 hover:to-amber-500 text-navy-950 font-bold px-8 py-4 rounded-xl transition-all duration-300 shadow-lg shadow-amber-500/25 hover:shadow-amber-500/40 hover:scale-[1.02]"
            >
              <span>Crea la tua squadra</span>
              <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
            </Link>
            <Link
              to="/artisti"
              className="inline-flex items-center space-x-2 bg-navy-800 hover:bg-navy-700 text-amber-600 dark:text-amber-400 border border-amber-500/30 dark:border-amber-400/30 font-semibold px-8 py-4 rounded-xl transition-all duration-300 hover:border-amber-500/50 dark:hover:border-amber-400/50"
            >
              <span>Esplora gli artisti</span>
              <Mic className="w-5 h-5" />
            </Link>
          </div>
        </div>
      </section>

      {/* Stats Cards */}
      <section className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatCard
          title="Artisti in gara"
          value="30"
          subtitle="Big 2026"
          icon={<Mic className="w-6 h-6" />}
          color="from-blue-500 to-cyan-500"
          delay={0}
        />
        <StatCard
          title="Budget disponibile"
          value="100"
          subtitle="Baudi"
          icon={<Coins className="w-6 h-6" />}
          color="from-amber-500 to-amber-500"
          delay={100}
        />
        <StatCard
          title="Artisti per squadra"
          value="7"
          subtitle="5 titolari + 2 riserve"
          icon={<Users className="w-6 h-6" />}
          color="from-purple-500 to-pink-500"
          delay={200}
        />
        <StatCard
          title="Date Festival"
          value="24-28 Feb"
          subtitle="2026"
          icon={<Calendar className="w-6 h-6" />}
          color="from-green-500 to-emerald-500"
          delay={300}
        />
      </section>

      {/* Quick Links */}
      <section>
        <h3 className="text-2xl font-bold text-amber-600 dark:text-amber-400 mb-6 flex items-center space-x-2">
          <Zap className="w-6 h-6" />
          <span>Accesso Rapido</span>
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <LinkCard
            to="/artisti"
            title="Lista Artisti"
            description="Visualizza tutti i 30 Big con quotazioni e predizioni ML"
            icon={<Mic className="w-8 h-8" />}
            delay={0}
          />
          <LinkCard
            to="/team-builder"
            title="Team Builder"
            description="Crea la tua squadra e simula il punteggio con l'algoritmo"
            icon={<Target className="w-8 h-8" />}
            delay={100}
          />
          <InfoCard
            title="Regolamento 2026"
            description="Budget 100 Baudi, 7 artisti (5 titolari + 2 riserve). Il capitano vale doppio!"
            icon={<Crown className="w-8 h-8" />}
            delay={200}
          />
        </div>
      </section>

      {/* Albo d'Oro */}
      <section className="card overflow-hidden">
        <div className="flex items-center space-x-3 mb-6">
          <div className="p-3 bg-gradient-to-br from-amber-500/20 to-amber-500/20 rounded-xl border border-amber-500/30 dark:border-amber-400/30">
            <Trophy className="w-6 h-6 text-amber-600 dark:text-amber-400" />
          </div>
          <div>
            <h3 className="text-2xl font-bold text-amber-600 dark:text-amber-400">Albo d'Oro</h3>
            <p className="text-sm text-gray-800 dark:text-gray-400">Storico vincitori del FantaSanremo</p>
          </div>
        </div>
        <div className="overflow-x-auto -mx-4 sm:mx-0 px-4 sm:px-0">
          <table className="w-full min-w-[600px]">
            <thead>
              <tr className="border-b border-navy-700">
                <th className="text-left py-4 px-4 text-gray-800 dark:text-gray-300 font-medium">Edizione</th>
                <th className="text-left py-4 px-4 text-gray-800 dark:text-gray-300 font-medium">Anno</th>
                <th className="text-left py-4 px-4 text-gray-800 dark:text-gray-300 font-medium">Squadre</th>
                <th className="text-left py-4 px-4 text-gray-800 dark:text-gray-300 font-medium">Vincitore</th>
                <th className="text-right py-4 px-4 text-gray-800 dark:text-gray-300 font-medium">Punteggio</th>
              </tr>
            </thead>
            <tbody>
              <AlboRow
                edizione="5ª"
                anno="2025"
                squadre="5.09M"
                vincitore="Olly"
                punteggio={475}
                trend="up"
              />
              <AlboRow
                edizione="4ª"
                anno="2024"
                squadre="4.20M"
                vincitore="La Sad"
                punteggio={486}
                trend="down"
              />
              <AlboRow
                edizione="3ª"
                anno="2023"
                squadre="4.21M"
                vincitore="Marco Mengoni"
                punteggio={670}
                highlight
                trend="record"
              />
              <AlboRow edizione="2ª" anno="2022" squadre="500K" vincitore="Emma" punteggio={525} trend="up" />
              <AlboRow
                edizione="1ª"
                anno="2021"
                squadre="47K"
                vincitore="Måneskin"
                punteggio={315}
                trend="up"
              />
            </tbody>
          </table>
        </div>
      </section>

      {/* ML Feature Highlight */}
      <section className="relative overflow-hidden rounded-3xl bg-gradient-to-br from-gray-100 to-gray-200 dark:from-navy-800 dark:to-navy-900 border border-gray-300 dark:border-navy-700">
        <div className="absolute top-0 right-0 w-64 h-64 bg-amber-400/5 blur-3xl rounded-full" />
        <div className="absolute bottom-0 left-0 w-64 h-64 bg-blue-400/5 blur-3xl rounded-full" />

        <div className="relative z-10 p-6 sm:p-8 md:p-12">
          <div className="flex flex-col md:flex-row items-center gap-8">
            <div className="flex-1 space-y-6">
              <div className="inline-flex items-center space-x-2 bg-blue-500/10 border border-blue-500/30 rounded-full px-4 py-2">
                <TrendingUp className="w-4 h-4 text-blue-600 dark:text-blue-400" />
                <span className="text-sm font-medium text-blue-600 dark:text-blue-400">Machine Learning</span>
              </div>
              <h3 className="text-3xl md:text-4xl font-bold text-amber-600 dark:text-amber-400">
                Predizioni basate su 50+ feature
              </h3>
              <p className="text-gray-800 dark:text-gray-200 text-lg leading-relaxed">
                Il nostro algoritmo analizza storico, genere, caratteristiche, bonus/malus e molto altro
                per aiutarti a scegliere la squadra vincente.
              </p>
              <Link
                to="/artisti"
                className="inline-flex items-center space-x-2 text-amber-600 dark:text-amber-400 hover:text-amber-700 dark:hover:text-amber-500 font-semibold transition-colors group"
              >
                <span>Scopri di più</span>
                <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
              </Link>
            </div>
            <div className="flex-shrink-0">
              <div className="relative">
                <div className="absolute inset-0 bg-gradient-to-r from-amber-400/20 to-blue-400/20 blur-2xl rounded-2xl" />
                <div className="relative bg-white/80 dark:bg-navy-800/80 backdrop-blur border border-gray-300 dark:border-navy-700 rounded-2xl p-8">
                  <div className="grid grid-cols-3 gap-4 text-center">
                    <div className="space-y-2">
                      <div className="text-2xl font-bold text-amber-600 dark:text-amber-400">50+</div>
                      <div className="text-xs text-gray-600 dark:text-gray-400">Feature</div>
                    </div>
                    <div className="space-y-2">
                      <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">108</div>
                      <div className="text-xs text-gray-600 dark:text-gray-400">Samples</div>
                    </div>
                    <div className="space-y-2">
                      <div className="text-2xl font-bold text-green-600 dark:text-green-400">5</div>
                      <div className="text-xs text-gray-600 dark:text-gray-400">Modelli</div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>
    </div>
  )
}

interface StatCardProps {
  title: string
  value: string
  subtitle: string
  icon: React.ReactNode
  color: string
  delay: number
}

function StatCard({ title, value, subtitle, icon, color, delay }: StatCardProps) {
  return (
    <div
      className="card group relative overflow-hidden transition-all duration-300 hover:scale-[1.02] hover:shadow-xl hover:shadow-amber-400/5"
      style={{ animationDelay: `${delay}ms` }}
    >
      {/* Background gradient on hover */}
      <div className={`absolute inset-0 bg-gradient-to-br ${color} opacity-0 group-hover:opacity-10 transition-opacity duration-300`} />

      {/* Icon container */}
      <div className="relative z-10 flex items-center justify-between mb-4">
        <div
          className={`p-3 bg-gradient-to-br ${color} rounded-xl bg-opacity-10 border border-white/10`}
        >
          <div className={`text-transparent bg-clip-text bg-gradient-to-br ${color}`}>
            {icon}
          </div>
        </div>
        <div className="text-3xl font-bold text-transparent bg-clip-text bg-gradient-to-br from-amber-600 to-amber-700 dark:from-amber-400 dark:to-amber-500">
          {value}
        </div>
      </div>

      {/* Content */}
      <div className="relative z-10">
        <div className="text-lg font-semibold text-gray-800 dark:text-gray-300 mb-1">{title}</div>
        <div className="text-sm text-gray-600 dark:text-gray-400">{subtitle}</div>
      </div>

      {/* Decorative element */}
      <div className={`absolute bottom-0 right-0 w-24 h-24 bg-gradient-to-br ${color} opacity-5 blur-2xl rounded-full group-hover:opacity-10 transition-opacity duration-300`} />
    </div>
  )
}

interface LinkCardProps {
  to: string
  title: string
  description: string
  icon: React.ReactNode
  delay: number
}

function LinkCard({ to, title, description, icon, delay }: LinkCardProps) {
  return (
    <Link
      to={to}
      className={`card group relative overflow-hidden transition-all duration-300 hover:scale-[1.02] hover:border-amber-400/50`}
      style={{ animationDelay: `${delay}ms` }}
    >
      {/* Hover gradient */}
      <div className="absolute inset-0 bg-gradient-to-br from-amber-400/0 via-amber-400/5 to-amber-400/0 opacity-0 group-hover:opacity-100 transition-opacity duration-300" />

      <div className="relative z-10">
        <div className="flex items-start justify-between mb-4">
          <div
            className="p-3 bg-gradient-to-br from-amber-500/10 to-amber-500/10 rounded-xl border border-amber-500/20 dark:border-amber-400/20 group-hover:border-amber-500/40 dark:group-hover:border-amber-400/40 transition-colors"
          >
            <div className="text-amber-600 dark:text-amber-400">{icon}</div>
          </div>
          <ArrowRight className="w-5 h-5 text-gray-600 dark:text-gray-400 group-hover:text-amber-400 group-hover:translate-x-1 transition-all" />
        </div>
        <h4 className="text-xl font-bold text-amber-600 dark:text-amber-400 mb-2 group-hover:text-amber-700 dark:group-hover:text-amber-500 transition-colors">
          {title}
        </h4>
        <p className="text-gray-700 dark:text-gray-300 text-sm leading-relaxed">{description}</p>
      </div>

      {/* Shimmer effect */}
      <div className="absolute inset-0 -translate-x-full group-hover:translate-x-full transition-transform duration-1000 ease-in-out bg-gradient-to-r from-transparent via-white/5 to-transparent" />
    </Link>
  )
}

interface InfoCardProps {
  title: string
  description: string
  icon: React.ReactNode
  delay: number
}

function InfoCard({ title, description, icon, delay }: InfoCardProps) {
  return (
    <div
      className="card group relative overflow-hidden transition-all duration-300"
      style={{ animationDelay: `${delay}ms` }}
    >
      <div className="relative z-10">
        <div className="flex items-start justify-between mb-4">
          <div
            className="p-3 bg-gradient-to-br from-purple-500/10 to-pink-500/10 rounded-xl border border-purple-500/20 dark:border-purple-400/20 group-hover:border-purple-500/40 dark:group-hover:border-purple-400/40 transition-colors"
          >
            <div className="text-purple-600 dark:text-purple-400">{icon}</div>
          </div>
        </div>
        <h4 className="text-xl font-bold text-purple-600 dark:text-purple-400 mb-2 group-hover:text-purple-700 dark:group-hover:text-purple-500 transition-colors">
          {title}
        </h4>
        <p className="text-gray-700 dark:text-gray-300 text-sm leading-relaxed">{description}</p>
      </div>
    </div>
  )
}

interface AlboRowProps {
  edizione: string
  anno: string
  squadre: string
  vincitore: string
  punteggio: number
  highlight?: boolean
  trend: 'up' | 'down' | 'record'
}

function AlboRow({ edizione, anno, squadre, vincitore, punteggio, highlight, trend }: AlboRowProps) {
  return (
    <tr
      className={`border-b border-gray-200 dark:border-navy-700 transition-all duration-200 hover:bg-gray-50 dark:hover:bg-navy-700/50 ${
        highlight ? 'bg-amber-50 dark:bg-amber-400/5 hover:bg-amber-100 dark:hover:bg-amber-400/10' : ''
      }`}
    >
      <td className="py-4 px-4">
        <span
          className={`inline-flex items-center justify-center w-10 h-10 rounded-lg font-bold ${
            highlight
              ? 'bg-amber-500/30 text-amber-800 dark:text-amber-400 border border-amber-500/40 dark:border-amber-400/30'
              : 'bg-gray-200 dark:bg-navy-800 text-gray-800 dark:text-gray-300'
          }`}
        >
          {edizione}
        </span>
      </td>
      <td className="py-4 px-4 text-gray-700 dark:text-gray-300">{anno}</td>
      <td className="py-4 px-4">
        <span className="inline-flex items-center space-x-1 text-gray-700 dark:text-gray-300">
          <Users className="w-4 h-4 text-amber-600 dark:text-amber-400" />
          <span>{squadre}</span>
        </span>
      </td>
      <td className="py-4 px-4">
        <div className="flex items-center space-x-2">
          {highlight && <Crown className="w-4 h-4 text-amber-600 dark:text-amber-400" />}
          <span className={`font-semibold ${highlight ? 'text-amber-600 dark:text-amber-400' : 'text-gray-700 dark:text-gray-300'}`}>
            {vincitore}
          </span>
        </div>
      </td>
      <td className="py-4 px-4 text-right">
        <div className="inline-flex items-center space-x-2">
          <span className="font-mono font-bold text-lg text-amber-600 dark:text-amber-400">{punteggio}</span>
          {trend === 'up' && <TrendingUp className="w-4 h-4 text-green-600 dark:text-green-400" />}
          {trend === 'down' && <TrendingUp className="w-4 h-4 text-red-600 dark:text-red-400 rotate-180" />}
          {trend === 'record' && (
            <span className="text-xs bg-green-500/20 text-green-600 dark:text-green-400 px-2 py-1 rounded-full">Record</span>
          )}
        </div>
      </td>
    </tr>
  )
}
