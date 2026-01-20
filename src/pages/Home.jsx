import { Link } from 'react-router-dom';
import { LayoutDashboard, Upload, Zap, BarChart3, Database } from 'lucide-react';
import TerminalMockup from "../components/TerminalMockup";

const Home = () => {
  return (
    <div className="space-y-24 pb-20">
      {/* Hero Section */}
      <section className="container mx-auto px-6 pt-20 flex flex-col lg:flex-row items-center justify-between gap-12">
        <div className="lg:w-1/2 space-y-8 text-left">
          <h1 className="text-6xl md:text-7xl font-display font-bold leading-tight">
            Turn data into <span className="text-primary">decisions.</span>
          </h1>
          <p className="text-xl text-text-muted max-w-lg leading-relaxed">
            The modern analytics platform for high-growth teams. Process millions of rows in milliseconds with AI-guided insights.
          </p>
          <div className="flex flex-wrap gap-4">
            <Link to="/upload" className="btn-primary text-lg px-8 py-3">
              Start Free Trial
            </Link>
            <Link to="/dashboard" className="btn-secondary text-lg px-8 py-3">
              View Demo
            </Link>
          </div>
          <div className="flex items-center gap-6 pt-4 grayscale opacity-60">
            <span className="text-sm font-medium text-text-dim uppercase tracking-wider">Trusted by</span>
            <div className="flex gap-8">
              <span className="font-display font-bold text-lg">Linear</span>
              <span className="font-display font-bold text-lg">Vercel</span>
              <span className="font-display font-bold text-lg">Supabase</span>
            </div>
          </div>
        </div>

        <div className="lg:w-1/2 relative">
          <TerminalMockup />
        </div>
      </section>

      {/* Stats Section */}
      <section className="bg-background-secondary py-12 border-y border-background-tertiary">
        <div className="container mx-auto px-6 grid grid-cols-2 md:grid-cols-4 gap-8">
          <div className="text-center">
            <div className="text-3xl font-display font-bold text-primary">10K+</div>
            <div className="text-sm text-text-muted mt-1">Datasets analyzed</div>
          </div>
          <div className="text-center">
            <div className="text-3xl font-display font-bold text-primary">50ms</div>
            <div className="text-sm text-text-muted mt-1">Avg query time</div>
          </div>
          <div className="text-center">
            <div className="text-3xl font-display font-bold text-primary">1M+</div>
            <div className="text-sm text-text-muted mt-1">Rows per second</div>
          </div>
          <div className="text-center">
            <div className="text-3xl font-display font-bold text-primary">99.9%</div>
            <div className="text-sm text-text-muted mt-1">Uptime SLA</div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="container mx-auto px-6 space-y-12">
        <div className="max-w-2xl">
          <h2 className="text-4xl font-bold">Everything you need to <br/>understand your data.</h2>
        </div>
        
        <div className="grid md:grid-cols-12 gap-6">
          <div className="md:col-span-7 glass p-8 rounded-xl space-y-6">
             <div className="w-10 h-10 bg-primary/10 rounded-lg flex items-center justify-center">
                <LayoutDashboard className="w-5 h-5 text-primary" />
             </div>
             <div className="space-y-2">
                <h3 className="text-2xl font-bold">Intelligent Dashboards</h3>
                <p className="text-text-muted">Analyze 1M+ rows in seconds. Our engine automatically generates the most relevant visualizations for your schema.</p>
             </div>
             <div className="h-48 bg-background-tertiary/30 rounded-lg border border-background-tertiary p-4 overflow-hidden font-mono text-xs">
                <code className="text-blue-400">SELECT</code> count(*), date_trunc('day', created_at)<br/>
                <code className="text-blue-400">FROM</code> user_events<br/>
                <code className="text-blue-400">GROUP BY</code> 2<br/>
                <code className="text-blue-400">ORDER BY</code> 2 <code className="text-blue-400">DESC</code>;
             </div>
          </div>

          <div className="md:col-span-5 glass p-8 rounded-xl space-y-6 flex flex-col justify-between">
             <div className="space-y-6">
                <div className="w-10 h-10 bg-primary/10 rounded-lg flex items-center justify-center">
                    <Zap className="w-5 h-5 text-primary" />
                </div>
                <div className="space-y-2">
                    <h3 className="text-2xl font-bold">AI Recommendations</h3>
                    <p className="text-text-muted">Get specific metrics and anomaly detection without writing a single query.</p>
                </div>
             </div>
             <div className="flex flex-col gap-2">
                <div className="flex items-center gap-3 bg-background-tertiary/50 p-3 rounded-lg border border-background-tertiary">
                    <div className="w-2 h-2 rounded-full bg-green-500"></div>
                    <span className="text-sm">Revenue up 12% vs last week</span>
                </div>
                <div className="flex items-center gap-3 bg-background-tertiary/50 p-3 rounded-lg border border-background-tertiary">
                    <div className="w-2 h-2 rounded-full bg-primary"></div>
                    <span className="text-sm">New segment detected: Enterprise</span>
                </div>
             </div>
          </div>
        </div>
      </section>
    </div>
  );
};

export default Home;
