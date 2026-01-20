import { useEffect, useState } from "react";
import { LayoutGrid, Loader2, TrendingUp, AlertTriangle, BarChart3, Target } from "lucide-react";
import { API_BASE_URL } from "../api";
import AIInsights from "../components/AIInsights";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
  LineChart, Line, PieChart, Pie, Cell, Legend
} from "recharts";

const COLORS = ['#6366F1', '#22D3EE', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6', '#EC4899', '#14B8A6'];

const Dashboard = () => {
  const [dashboard, setDashboard] = useState(null);
  const [loading, setLoading] = useState(true);
  const [mlStatus, setMlStatus] = useState(null);

  useEffect(() => {
    // Fetch dashboard data
    fetch(`${API_BASE_URL}/api/dashboard`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ fields: [] })
    })
      .then(res => res.json())
      .then(data => {
        setDashboard(data);
        setMlStatus(data.ml_status);
        setLoading(false);
      });
  }, []);

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center py-20 gap-4">
        <Loader2 className="w-10 h-10 animate-spin text-primary" />
        <p className="text-text-muted">Loading dashboard...</p>
      </div>
    );
  }

  const renderChartByType = (chart, index) => {
    const chartData = chart.data || [];
    
    if (chart.recommended_chart === 'line' || chart.title?.toLowerCase().includes('trend')) {
      return (
        <LineChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" stroke="#333" />
          <XAxis dataKey="name" fontSize={11} />
          <YAxis fontSize={11} />
          <Tooltip formatter={(v) => v.toLocaleString()} />
          <Line type="monotone" dataKey="value" stroke="#6366F1" strokeWidth={2} dot={{ fill: '#6366F1' }} />
        </LineChart>
      );
    }
    
    if (chart.recommended_chart === 'pie' || chart.title?.toLowerCase().includes('distribution')) {
      return (
        <PieChart>
          <Pie
            data={chartData}
            dataKey="value"
            nameKey="name"
            cx="50%"
            cy="50%"
            outerRadius={80}
            label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
          >
            {chartData.map((entry, i) => (
              <Cell key={`cell-${i}`} fill={COLORS[i % COLORS.length]} />
            ))}
          </Pie>
          <Tooltip formatter={(v) => v.toLocaleString()} />
          <Legend />
        </PieChart>
      );
    }
    
    // Default: Bar chart
    return (
      <BarChart data={chartData}>
        <defs>
          <linearGradient id={`color${index}`} x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="#6366F1" stopOpacity={0.9}/>
            <stop offset="95%" stopColor="#22D3EE" stopOpacity={0.7}/>
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke="#333" />
        <XAxis dataKey="name" angle={-45} textAnchor="end" height={80} fontSize={10} />
        <YAxis fontSize={11} />
        <Tooltip formatter={(v) => v.toLocaleString()} />
        <Bar dataKey="value" fill={`url(#color${index})`} radius={[8,8,0,0]} />
      </BarChart>
    );
  };

  return (
    <div className="max-w-7xl mx-auto px-6 py-20">
      <h1 className="text-4xl mb-6 flex items-center gap-3">
        <LayoutGrid className="text-primary" /> Analytics Dashboard
      </h1>

      {/* Dataset Info & ML Status */}
      <div className="flex flex-wrap gap-4 mb-8">
        {dashboard.dataset && (
          <div className="bg-background-secondary px-4 py-2 rounded-lg border">
            <span className="text-text-muted">Dataset:</span> <span className="font-semibold">{dashboard.dataset}</span>
          </div>
        )}
        <div className={`px-4 py-2 rounded-lg flex items-center gap-2 ${
          mlStatus === 'completed' ? 'bg-green-500/20 text-green-400 border border-green-500/30' : 
          'bg-yellow-500/20 text-yellow-400 border border-yellow-500/30'
        }`}>
          {mlStatus === 'completed' ? <TrendingUp className="w-4 h-4" /> : <Loader2 className="w-4 h-4 animate-spin" />}
          <span>ML Analysis: {mlStatus === 'completed' ? 'Complete' : 'Processing'}</span>
        </div>
      </div>

      {/* Key Findings Section */}
      {dashboard.key_findings && dashboard.key_findings.length > 0 && (
        <div className="mb-10">
          <h2 className="text-2xl mb-4 flex items-center gap-2">
            <Target className="text-primary" /> Key Findings
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {dashboard.key_findings.map((finding, i) => (
              <div key={i} className="bg-background-secondary border rounded-xl p-5">
                <h3 className="text-lg font-semibold mb-2 flex items-center gap-2">
                  {finding.type === 'intervention' ? <AlertTriangle className="w-5 h-5 text-orange-500" /> : 
                   finding.type === 'models' ? <BarChart3 className="w-5 h-5 text-blue-500" /> :
                   <TrendingUp className="w-5 h-5 text-green-500" />}
                  {finding.title}
                </h3>
                
                {finding.type === 'models' && finding.data && (
                  <div className="space-y-2 text-sm">
                    {Object.entries(finding.data).map(([model, metrics]) => (
                      <div key={model} className="flex justify-between">
                        <span className="text-text-muted">{model}:</span>
                        <span>MAE: {metrics.mae?.toFixed(2) || 'N/A'}</span>
                      </div>
                    ))}
                  </div>
                )}
                
                {finding.type === 'intervention' && finding.areas && (
                  <ul className="text-sm space-y-1">
                    {finding.areas.slice(0, 5).map((area, j) => (
                      <li key={j} className="flex justify-between">
                        <span className="text-text-muted truncate">{area.district}</span>
                        <span className="text-orange-400">{area.severity_score?.toFixed(1)}</span>
                      </li>
                    ))}
                  </ul>
                )}
                
                {finding.type === 'states' && finding.top_states && (
                  <ul className="text-sm space-y-1">
                    {finding.top_states.slice(0, 5).map((state, j) => (
                      <li key={j} className="flex justify-between">
                        <span className="text-text-muted">{state.state || state.name}</span>
                        <span className="text-green-400">{(state.total || state.value)?.toLocaleString()}</span>
                      </li>
                    ))}
                  </ul>
                )}
                
                {finding.type === 'forecast' && finding.data && (
                  <div className="text-sm space-y-2">
                    {/* Prophet forecasts */}
                    {finding.data.prophet?.predictions && finding.data.prophet.predictions.length > 0 && (
                      <div>
                        <span className="text-text-muted">Prophet ({finding.data.prophet.trend || 'forecast'}):</span>
                        <ul className="mt-1 space-y-1">
                          {finding.data.prophet.predictions.filter(f => f.predicted > 0).slice(0, 3).map((f, j) => (
                            <li key={j} className="flex justify-between">
                              <span className="text-blue-400">{f.month}</span>
                              <span className="text-green-400">{Math.round(f.predicted)?.toLocaleString()}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}
                    {/* ARIMA forecasts */}
                    {finding.data.arima?.predictions && finding.data.arima.predictions.length > 0 && (
                      <div className="mt-2">
                        <span className="text-text-muted">ARIMA:</span>
                        <ul className="mt-1 space-y-1">
                          {finding.data.arima.predictions.filter(f => f.predicted > 0).slice(0, 3).map((f, j) => (
                            <li key={j} className="flex justify-between">
                              <span className="text-purple-400">{f.month}</span>
                              <span className="text-green-400">{Math.round(f.predicted)?.toLocaleString()}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}
                    {/* LSTM forecasts */}
                    {finding.data.lstm?.predictions && finding.data.lstm.predictions.length > 0 && (
                      <div className="mt-2">
                        <span className="text-text-muted">LSTM:</span>
                        <ul className="mt-1 space-y-1">
                          {finding.data.lstm.predictions.filter(f => f.predicted > 0).slice(0, 3).map((f, j) => (
                            <li key={j} className="flex justify-between">
                              <span className="text-cyan-400">{f.month}</span>
                              <span className="text-green-400">{Math.round(f.predicted)?.toLocaleString()}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Charts Section */}
      {dashboard.charts && dashboard.charts.length > 0 && (
        <>
          <h2 className="text-2xl mb-4 flex items-center gap-2">
            <BarChart3 className="text-primary" /> Visualizations
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-10 mb-10">
            {dashboard.charts.map((chart, i) => (
              <div key={i} className="bg-background-secondary border rounded-2xl p-6 shadow-xl">
                <h2 className="text-xl mb-2">{chart.title || `${chart.field} Distribution`}</h2>
                {chart.insight && <p className="text-text-muted text-sm mb-4">{chart.insight}</p>}

                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    {renderChartByType(chart, i)}
                  </ResponsiveContainer>
                </div>
              </div>
            ))}
          </div>
        </>
      )}

      {/* AI Insights Section */}
      {dashboard.analysis?.auto_generated_insights && dashboard.analysis.auto_generated_insights.length > 0 && (
        <AIInsights insights={dashboard.analysis.auto_generated_insights} />
      )}
    </div>
  );
};

export default Dashboard;
