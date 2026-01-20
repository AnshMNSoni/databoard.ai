import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import Home from './pages/Home'
import Upload from './pages/Upload'
import Schema from './pages/Schema'
import Analyze from './pages/Analyze'
import Suggestions from './pages/Suggestions'
import Dashboard from './pages/Dashboard'
import Report from './pages/Report'
import Navbar from './components/Navbar'
import AnimatedBackground from './components/AnimatedBackground'

function App() {
  return (
    <Router>
      <div className="min-h-screen noise-bg">
        <AnimatedBackground />
        <div className="fixed inset-0 grid-pattern pointer-events-none opacity-50" />
        <div className="relative z-10">
          <Navbar />
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/upload" element={<Upload />} />
            <Route path="/schema" element={<Schema />} />
            <Route path="/analyze" element={<Analyze />} />
            <Route path="/suggestions" element={<Suggestions />} />
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/report" element={<Report />} />
          </Routes>
        </div>
      </div>
    </Router>
  )
}

export default App
