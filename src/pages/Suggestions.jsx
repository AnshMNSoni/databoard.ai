import { useEffect, useState } from "react";
import { Lightbulb, Loader2, Sparkles, TrendingUp, Info } from "lucide-react";
import { API_BASE_URL } from "../api";

import {
  BarChart, Bar,
  LineChart, Line,
  PieChart, Pie,
  Tooltip, XAxis, YAxis,
  ResponsiveContainer, Legend, Cell, CartesianGrid
} from "recharts";

const COLORS = ['#6366F1', '#22D3EE', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6', '#EC4899', '#14B8A6'];

const Suggestions = () => {
  const [suggestions, setSuggestions] = useState([]);
  const [aiRecommendations, setAiRecommendations] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch(`${API_BASE_URL}/api/suggest`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ fields: [] })
    })
      .then(res => res.json())
      .then(data => {
        setSuggestions(data.suggestions || []);
        setAiRecommendations(data.ai_recommendations || []);
        setLoading(false);
      })
      .catch(err => {
        console.error("Failed to load suggestions:", err);
        setLoading(false);
      });
  }, []);

  const renderChart = (chart, index) => {
    if (!chart.data || chart.data.length === 0) return null;

    if (chart.recommended_chart === "bar") {
      return (
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={chart.data}>
            <CartesianGrid strokeDasharray="3 3" stroke="#333" />
            <XAxis dataKey="name" angle={-45} textAnchor="end" height={80} fontSize={11} />
            <YAxis fontSize={11} />
            <Tooltip formatter={(v) => v.toLocaleString()} />
            <Bar dataKey="value" radius={[8,8,0,0]}>
              {chart.data.map((entry, i) => (
                <Cell key={`cell-${i}`} fill={COLORS[i % COLORS.length]} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      );
    }

    if (chart.recommended_chart === "line") {
      return (
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={chart.data}>
            <CartesianGrid strokeDasharray="3 3" stroke="#333" />
            <XAxis dataKey="name" fontSize={11} />
            <YAxis fontSize={11} />
            <Tooltip formatter={(v) => v.toLocaleString()} />
            <Line type="monotone" dataKey="value" stroke="#6366F1" strokeWidth={2} dot={{ fill: '#6366F1' }} />
          </LineChart>
        </ResponsiveContainer>
      );
    }

    if (chart.recommended_chart === "pie") {
      return (
        <ResponsiveContainer width="100%" height={300}>
          <PieChart>
            <Pie
              data={chart.data}
              dataKey="value"
              nameKey="name"
              cx="50%"
              cy="50%"
              outerRadius={100}
              label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
            >
              {chart.data.map((entry, i) => (
                <Cell key={`cell-${i}`} fill={COLORS[i % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip formatter={(v) => v.toLocaleString()} />
            <Legend />
          </PieChart>
        </ResponsiveContainer>
      );
    }

    return null;
  };

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center py-20 gap-4">
        <Loader2 className="w-10 h-10 animate-spin text-primary" />
        <p className="text-text-muted">Generating AI suggestions...</p>
      </div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto px-6 py-20">
      <div className="flex items-center gap-3 mb-4">
        <Lightbulb className="w-8 h-8 text-primary" />
        <h1 className="text-4xl">AI Chart Suggestions</h1>
      </div>
      
      <p className="text-text-muted mb-8">
        Smart visualization recommendations based on your Aadhaar enrollment data analysis.
      </p>

      {/* AI Recommendations Panel */}
      {aiRecommendations.length > 0 && (
        <div className="bg-gradient-to-r from-primary/10 to-purple-500/10 p-6 rounded-xl border border-primary/30 mb-10">
          <h2 className="text-xl mb-4 flex items-center gap-2">
            <Sparkles className="w-5 h-5 text-primary" />
            AI-Powered Recommendations
          </h2>
          <ul className="space-y-2">
            {aiRecommendations.map((rec, i) => (
              <li key={i} className="flex items-start gap-3 p-2 bg-background/30 rounded-lg">
                <TrendingUp className="w-4 h-4 text-green-400 mt-1 flex-shrink-0" />
                <span>{rec}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Suggested Charts */}
      <div className="grid gap-8">
        {suggestions.map((s, i) => (
          <div
            key={i}
            className="p-6 bg-background-secondary rounded-xl border shadow-lg"
          >
            <div className="flex items-start justify-between mb-4">
              <div>
                <h2 className="text-xl font-semibold">
                  {s.title || `${s.field.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())} Analysis`}
                </h2>
                <p className="text-text-muted text-sm mt-1">
                  Recommended: <span className="text-primary font-medium">{s.recommended_chart?.toUpperCase()}</span> chart
                </p>
              </div>
              <span className="px-3 py-1 bg-primary/20 text-primary text-sm rounded-full">
                {s.field}
              </span>
            </div>

            {/* Insight */}
            {s.insight && (
              <div className="flex items-center gap-2 mb-4 p-3 bg-blue-500/10 rounded-lg border border-blue-500/20">
                <Info className="w-4 h-4 text-blue-400 flex-shrink-0" />
                <span className="text-sm text-blue-300">{s.insight}</span>
              </div>
            )}

            {renderChart(s, i)}
          </div>
        ))}

        {suggestions.length === 0 && (
          <div className="text-center py-10 text-text-muted">
            <Lightbulb className="w-12 h-12 mx-auto mb-4 opacity-50" />
            <p>No suggestions available. Upload a dataset to get started.</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default Suggestions;
