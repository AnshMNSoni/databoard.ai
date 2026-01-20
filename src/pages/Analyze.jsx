import { useEffect, useState } from "react";
import { Settings2, Loader2, Sparkles, TrendingUp, AlertCircle } from "lucide-react";
import { API_BASE_URL } from "../api";
import { getChartConfigFromGemini } from "../gemini";
import { BarChart, Bar, LineChart, Line, PieChart, Pie, Tooltip, XAxis, YAxis, ResponsiveContainer, Cell, Legend } from "recharts";

const COLORS = ['#8884d8', '#82ca9d', '#ffc658', '#ff7300', '#0088fe', '#00C49F', '#FFBB28', '#FF8042', '#a4de6c', '#d0ed57'];

const Analyze = () => {
  const [fields, setFields] = useState([]);
  const [selectedFields, setSelectedFields] = useState([]);
  const [loading, setLoading] = useState(false);
  const [chartConfig, setChartConfig] = useState(null);
  const [aiInsights, setAiInsights] = useState([]);
  const [mlStatus, setMlStatus] = useState(null);

  useEffect(() => {
    fetch(`${API_BASE_URL}/api/schema`)
      .then(res => res.json())
      .then(data => {
        if (data.schema) setFields(data.schema.map(col => col.name));
      });
      
    // Check ML status
    fetch(`${API_BASE_URL}/api/ml/results`)
      .then(res => res.json())
      .then(data => {
        setMlStatus(data.status);
      })
      .catch(() => setMlStatus('unavailable'));
  }, []);

  const toggleField = (field) => {
    setSelectedFields(prev =>
      prev.includes(field)
        ? prev.filter(f => f !== field)
        : [...prev, field]
    );
  };

  const runAnalysis = async () => {
    if (selectedFields.length === 0) return;
    setLoading(true);

    try {
      const res = await fetch(`${API_BASE_URL}/api/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ fields: selectedFields })
      });

      const data = await res.json();
      
      // Store AI insights from backend
      if (data.ai_insights) {
        setAiInsights(data.ai_insights);
      }

      // Send analysis data along with AI insights to Gemini for chart generation
      const chart = await getChartConfigFromGemini(data.analysis, data.ai_insights || []);
      setChartConfig(chart);
      
      // Merge insights from chart config if available
      if (chart.insights && chart.insights.length > 0) {
        setAiInsights(prev => {
          const combined = [...new Set([...prev, ...chart.insights])];
          return combined.slice(0, 10);
        });
      }
    } catch (error) {
      console.error("Analysis failed:", error);
    }

    setLoading(false);
  };

  const renderChart = () => {
    if (!chartConfig) return null;

    if (chartConfig.type === "bar") {
      return (
        <BarChart data={chartConfig.data}>
          <XAxis dataKey="name" angle={-45} textAnchor="end" height={80} fontSize={12} />
          <YAxis />
          <Tooltip formatter={(value) => value.toLocaleString()} />
          <Legend />
          <Bar dataKey="value" fill="#8884d8">
            {chartConfig.data.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
            ))}
          </Bar>
        </BarChart>
      );
    }

    if (chartConfig.type === "line") {
      return (
        <LineChart data={chartConfig.data}>
          <XAxis dataKey="name" />
          <YAxis />
          <Tooltip formatter={(value) => value.toLocaleString()} />
          <Legend />
          <Line type="monotone" dataKey="value" stroke="#8884d8" strokeWidth={2} dot={{ fill: '#8884d8' }} />
        </LineChart>
      );
    }

    if (chartConfig.type === "pie") {
      return (
        <PieChart>
          <Pie 
            data={chartConfig.data} 
            dataKey="value" 
            nameKey="name" 
            cx="50%" 
            cy="50%" 
            outerRadius={120}
            label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(1)}%`}
          >
            {chartConfig.data.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
            ))}
          </Pie>
          <Tooltip formatter={(value) => value.toLocaleString()} />
          <Legend />
        </PieChart>
      );
    }
  };

  return (
    <div className="max-w-6xl mx-auto px-6 py-20">
      <h1 className="text-4xl mb-6 flex items-center gap-3">
        <Sparkles className="w-10 h-10 text-primary" />
        AI-Powered Analysis
      </h1>

      {/* ML Status Indicator */}
      {mlStatus && (
        <div className={`mb-4 p-3 rounded-lg flex items-center gap-2 ${
          mlStatus === 'completed' ? 'bg-green-500/20 text-green-400' : 
          mlStatus === 'processing' ? 'bg-yellow-500/20 text-yellow-400' : 
          'bg-gray-500/20 text-gray-400'
        }`}>
          {mlStatus === 'completed' ? <TrendingUp className="w-5 h-5" /> : <AlertCircle className="w-5 h-5" />}
          <span>ML Analysis: {mlStatus === 'completed' ? 'Ready - Enhanced insights available' : mlStatus}</span>
        </div>
      )}

      <div className="bg-background-secondary p-8 rounded-xl border mb-10">
        <h2 className="text-xl mb-4">Select Fields for Analysis</h2>
        <p className="text-text-muted mb-4">Choose the data fields you want to analyze. Our AI will generate insights and visualizations.</p>

        <div className="grid grid-cols-2 md:grid-cols-3 gap-4 mb-8">
          {fields.map((field, i) => (
            <label key={i} className={`flex items-center gap-2 p-3 rounded-lg cursor-pointer transition-all ${
              selectedFields.includes(field) 
                ? 'bg-primary/20 border border-primary' 
                : 'bg-background hover:bg-background-tertiary border border-transparent'
            }`}>
              <input
                type="checkbox"
                checked={selectedFields.includes(field)}
                onChange={() => toggleField(field)}
                className="accent-primary"
              />
              <span className="capitalize">{field.replace(/_/g, ' ')}</span>
            </label>
          ))}
        </div>

        <button
          onClick={runAnalysis}
          disabled={loading || selectedFields.length === 0}
          className="px-8 py-3 bg-primary text-white rounded-lg flex items-center gap-2 hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
        >
          {loading && <Loader2 className="w-4 h-4 animate-spin" />}
          {loading ? 'Analyzing...' : 'Run AI Analysis'}
        </button>
      </div>

      {/* Chart Visualization */}
      {chartConfig && (
        <div className="bg-background-secondary p-8 rounded-xl border mb-10">
          <h2 className="text-2xl mb-4">{chartConfig.title}</h2>

          <ResponsiveContainer width="100%" height={400}>
            {renderChart()}
          </ResponsiveContainer>
        </div>
      )}

      {/* AI Insights Panel */}
      {aiInsights.length > 0 && (
        <div className="bg-gradient-to-r from-primary/10 to-purple-500/10 p-8 rounded-xl border border-primary/30">
          <h2 className="text-2xl mb-4 flex items-center gap-2">
            <Sparkles className="w-6 h-6 text-primary" />
            AI-Generated Insights
          </h2>
          
          <ul className="space-y-3">
            {aiInsights.map((insight, i) => (
              <li key={i} className="flex items-start gap-3 p-3 bg-background/50 rounded-lg">
                <span className="bg-primary text-white text-sm font-bold w-6 h-6 rounded-full flex items-center justify-center flex-shrink-0">
                  {i + 1}
                </span>
                <span className="text-text">{insight}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

export default Analyze;
