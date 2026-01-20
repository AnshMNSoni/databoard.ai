export async function getChartConfigFromGemini(analysisData, aiInsights = []) {
  // Control Gemini usage - set to true when you have valid API key
  const USE_GEMINI = true; 
  
  const apiKey = import.meta.env.VITE_GEMINI_API_KEY;

  // Prepare analysis text for both Gemini and fallback
  const analysisText = typeof analysisData === 'string' ? analysisData : JSON.stringify(analysisData);

  if (!apiKey || !USE_GEMINI) {
    console.log("Using fallback chart (Gemini disabled or no API key)");
    return getFallbackChart(analysisText, aiInsights);
  }

  try {
    const prompt = `You are an expert data analyst for Aadhaar enrollment data in India.
    
Given this analytics result, generate:
1. Best chart configuration for visualization
2. AI-powered insights explaining the data

Return ONLY valid JSON in this exact format:
{
  "type": "bar" | "line" | "pie",
  "title": "Chart title describing the visualization",
  "data": [{ "name": "label", "value": number }],
  "insights": [
    "Insight 1: Key finding from the data",
    "Insight 2: Important trend or pattern",
    "Insight 3: Actionable recommendation"
  ]
}

Analysis Data:
${analysisText}

${aiInsights.length > 0 ? `ML-Generated Insights: ${aiInsights.join('; ')}` : ''}`;

    const response = await fetch(
      `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=${apiKey}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          contents: [{ parts: [{ text: prompt }] }]
        })
      }
    );

    if (!response.ok) {
      console.error("Gemini API error:", response.status, await response.text());
      return getFallbackChart(analysisText, aiInsights);
    }

    const data = await response.json();
    
    if (!data.candidates || !data.candidates[0]?.content?.parts?.[0]?.text) {
      console.error("Invalid Gemini response:", data);
      return getFallbackChart(analysisText, aiInsights);
    }

    let text = data.candidates[0].content.parts[0].text;
    // Remove markdown code blocks if present
    text = text.replace(/```json\n?/g, '').replace(/```\n?/g, '').trim();
    
    return JSON.parse(text);
  } catch (error) {
    console.error("Gemini API failed:", error);
    return getFallbackChart(analysisText, aiInsights);
  }
}

// Fallback chart when Gemini fails or is disabled
function getFallbackChart(analysisText, aiInsights = []) {
  try {
    console.log("Fallback chart - processing analysis");
    const analysis = typeof analysisText === 'string' ? JSON.parse(analysisText) : analysisText;
    
    // Find the best field to visualize
    const fields = Object.keys(analysis).filter(k => !['summary', 'ai_insights', 'ml_available'].includes(k));
    
    if (fields.length === 0) {
      return getDefaultChart(aiInsights);
    }
    
    const firstField = fields[0];
    const fieldData = analysis[firstField];
    
    console.log("Processing field:", firstField, fieldData);
    
    // Generate insights from analysis and AI
    const insights = generateInsights(fieldData, firstField, aiInsights);
    
    // Handle Aadhaar-specific data formats
    if (fieldData?.top_states) {
      return {
        type: "bar",
        title: `Top States by ${firstField.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}`,
        data: fieldData.top_states.slice(0, 10),
        insights: insights
      };
    }
    
    if (fieldData?.top_values) {
      return {
        type: "bar",
        title: `${firstField.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())} Distribution`,
        data: fieldData.top_values.slice(0, 10),
        insights: insights
      };
    }
    
    if (fieldData?.monthly_trend) {
      return {
        type: "line",
        title: "Enrollment Trend Over Time",
        data: fieldData.monthly_trend,
        insights: insights
      };
    }
    
    if (fieldData?.top_districts) {
      return {
        type: "bar",
        title: "Top Districts by Activity",
        data: fieldData.top_districts.slice(0, 10),
        insights: insights
      };
    }
    
    if (fieldData?.total_enrollments !== undefined) {
      // Age group data
      return {
        type: "pie",
        title: fieldData.age_group || `${firstField} Enrollment`,
        data: fieldData.top_states || [{ name: "Total", value: fieldData.total_enrollments }],
        insights: insights
      };
    }
    
    // Handle generic object data (value counts)
    if (typeof fieldData === 'object' && !Array.isArray(fieldData)) {
      const entries = Object.entries(fieldData)
        .filter(([k]) => !['insight', 'total_states', 'total_districts', 'total_pincodes'].includes(k))
        .slice(0, 10);
      
      if (entries.length > 0) {
        return {
          type: "bar",
          title: `${firstField.replace(/_/g, ' ')} Analysis`,
          data: entries.map(([name, value]) => ({
            name: String(name).substring(0, 20),
            value: typeof value === 'number' ? value : 0
          })),
          insights: insights
        };
      }
    }
    
    return getDefaultChart(aiInsights);
    
  } catch (e) {
    console.error("Fallback parsing failed:", e);
    return getDefaultChart(aiInsights);
  }
}

function generateInsights(fieldData, fieldName, aiInsights) {
  const insights = [];
  
  // Add AI insights first
  if (aiInsights && aiInsights.length > 0) {
    insights.push(...aiInsights.slice(0, 3));
  }
  
  // Add field-specific insights
  if (fieldData?.insight) {
    insights.push(fieldData.insight);
  }
  
  if (fieldData?.total_enrollments) {
    insights.push(`Total enrollments: ${fieldData.total_enrollments.toLocaleString()}`);
  }
  
  if (fieldData?.total_states) {
    insights.push(`Covers ${fieldData.total_states} states/UTs`);
  }
  
  if (fieldData?.total_districts) {
    insights.push(`Active in ${fieldData.total_districts} districts`);
  }
  
  return insights.length > 0 ? insights : ["Analysis completed successfully"];
}

function getDefaultChart(aiInsights) {
  return {
    type: "bar",
    title: "Analysis Results",
    data: [{ name: "No visualization data", value: 0 }],
    insights: aiInsights.length > 0 ? aiInsights : ["Upload data and select fields to analyze"]
  };
}
