import { Database, AlertCircle, ArrowUp, Loader2 } from "lucide-react";
import { Link } from "react-router-dom";
import { useEffect, useState } from "react";
import { API_BASE_URL } from "../api";

const Schema = () => {
  const [schema, setSchema] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchSchema = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/api/schema`);
        const data = await response.json();

        if (!response.ok) {
          throw new Error(data.error || "Failed to fetch schema");
        }

        if (data.schema) {
          setSchema(data.schema);
        } else {
          setSchema([]);
        }
      } catch (err) {
        setError(err.message || "Failed to load schema");
      } finally {
        setLoading(false);
      }
    };

    fetchSchema();
  }, []);

  return (
    <div className="max-w-5xl mx-auto px-6 py-20">
      <div className="mb-12">
        <h1 className="text-4xl mb-4">Schema Detection</h1>
        <p className="text-text-muted">
          Auto-detected columns and data types from your dataset
        </p>
      </div>

      <div className="relative border-2 border-dashed border-background-tertiary bg-background-secondary rounded-2xl p-12 transition-all">

        {/* Loading */}
        {loading && (
          <div className="flex flex-col items-center justify-center py-20">
            <Loader2 className="w-10 h-10 animate-spin text-primary mb-4" />
            <p className="text-text-muted">Detecting schema...</p>
          </div>
        )}

        {/* Error */}
        {!loading && error && (
          <div className="flex flex-col items-center justify-center py-20">
            <div className="w-16 h-16 rounded-full bg-red-500/10 text-red-500 flex items-center justify-center mb-6">
              <AlertCircle className="w-8 h-8" />
            </div>
            <p className="text-red-400 mb-6">{error}</p>
            <Link
              to="/upload"
              className="px-8 py-3 bg-primary hover:bg-primary-dark text-white rounded-lg font-semibold transition-all flex items-center gap-2"
            >
              <ArrowUp className="w-4 h-4" /> Upload Dataset
            </Link>
          </div>
        )}

        {/* No Dataset */}
        {!loading && !error && schema.length === 0 && (
          <div className="flex flex-col items-center justify-center py-20">
            <div className="w-16 h-16 rounded-full bg-orange-500/10 text-orange-500 flex items-center justify-center mb-6">
              <Database className="w-8 h-8" />
            </div>
            <p className="text-text-muted mb-6">
              No dataset uploaded yet. Please upload a dataset to view schema.
            </p>
            <Link
              to="/upload"
              className="px-8 py-3 bg-primary hover:bg-primary-dark text-white rounded-lg font-semibold transition-all flex items-center gap-2"
            >
              <ArrowUp className="w-4 h-4" /> Go to Upload Page
            </Link>
          </div>
        )}

        {/* Schema Table */}
        {!loading && !error && schema.length > 0 && (
          <div>
            <div className="flex items-center gap-3 mb-6">
              <Database className="w-6 h-6 text-primary" />
              <h2 className="text-2xl font-semibold">Detected Schema</h2>
            </div>

            <div className="overflow-x-auto rounded-xl border border-background-tertiary">
              <table className="w-full text-left border-collapse">
                <thead className="bg-background-tertiary">
                  <tr>
                    <th className="px-6 py-3 text-sm font-semibold">Column Name</th>
                    <th className="px-6 py-3 text-sm font-semibold">Data Type</th>
                  </tr>
                </thead>
                <tbody>
                  {schema.map((col, index) => (
                    <tr
                      key={index}
                      className="border-t border-background-tertiary hover:bg-background-tertiary/50 transition-colors"
                    >
                      <td className="px-6 py-3">{col.name}</td>
                      <td className="px-6 py-3 text-text-muted">{col.type}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className="mt-10 flex justify-end">
              <Link
                to="/analyze"
                className="px-8 py-3 bg-primary hover:bg-primary-dark text-white rounded-lg font-semibold transition-all flex items-center gap-2"
              >
                Continue to Analyze
              </Link>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default Schema;
