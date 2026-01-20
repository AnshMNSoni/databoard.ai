import { Link, useLocation } from 'react-router-dom'
import { clsx } from 'clsx'

const Navbar = () => {
  const location = useLocation()
  
  const navItems = [
    { name: 'Upload', path: '/upload' },
    { name: 'Schema', path: '/schema' },
    { name: 'Analyze', path: '/analyze' },
    { name: 'Suggestions', path: '/suggestions' },
    { name: 'Dashboard', path: '/dashboard' },
    { name: 'Report', path: '/report' },
  ]

  return (
    <nav className="border-b border-background-tertiary/50 bg-background/80 backdrop-blur-md sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
        <Link to="/" className="text-xl font-bold tracking-tighter">
          Databoard<span className="text-primary">.ai</span>
        </Link>
        
        <div className="flex gap-4 lg:gap-8 items-center">
          {navItems.map((item) => (
            <Link
              key={item.path}
              to={item.path}
              className={clsx(
                "text-xs lg:text-sm font-medium transition-all px-3 py-1.5 rounded-lg",
                location.pathname === item.path 
                  ? "bg-blue-600 text-white shadow-lg shadow-blue-600/20" 
                  : "text-text-muted hover:text-text"
              )}
            >
              {item.name}
            </Link>
          ))}
        </div>
      </div>
    </nav>
  )
}

export default Navbar
