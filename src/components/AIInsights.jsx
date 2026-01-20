import React from "react";

const AIInsights = ({ insights }) => {
  if (!insights || insights.length === 0) return null;

  return (
    <div className="bg-background-secondary border rounded-2xl p-8 shadow-xl mt-10">
      <h2 className="text-2xl mb-6">AI Insights Report</h2>

      <div className="space-y-6 text-left font-serif" style={{ fontFamily: "Times New Roman" }}>
        {insights.map((insight, index) => (
          <div key={index}>
            <h3 className="text-lg font-bold mb-2">
              {index + 1}. {extractHeading(insight)}
            </h3>

            <ul className="list-disc ml-6 text-text-muted space-y-1">
              {extractPoints(insight).map((point, i) => (
                <li key={i}>{point}</li>
              ))}
            </ul>
          </div>
        ))}
      </div>
    </div>
  );
};

const extractHeading = (text) => {
  const parts = text.split(":");
  return parts[0].replace(/\*\*/g, "").trim();
};

const extractPoints = (text) => {
  const parts = text.split(":");
  if (parts.length < 2) return [text];

  return parts[1]
    .replace(/\*\*/g, "")
    .split(".")
    .map(p => p.trim())
    .filter(p => p.length > 5);
};

export default AIInsights;
