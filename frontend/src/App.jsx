import React, { useState, useEffect } from 'react';
import { ShieldAlert, Activity, Cpu, Play, Radio } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts';

export default function App() {
  const [recon, setRecon] = useState(0.6);
  const [lateral, setLateral] = useState(0.2);
  const [exfil, setExfil] = useState(0.3);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);

  // Replace with your active backend URL (e.g. Render / Cloud Run / localhost:8000)
  const BACKEND_URL = "http://localhost:8000/api/forecast";

  const runSimulation = async () => {
    setLoading(true);
    try {
      const res = await fetch(BACKEND_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ recon_intensity: recon, lateral_activity: lateral, exfil_volume: exfil, horizon_k: 6 })
      });
      const data = await res.json();
      setResults(data);
    } catch (err) {
      console.error("Simulation request failed", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    runSimulation();
  }, []);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 p-8 font-sans">
      <header className="max-w-7xl mx-auto flex items-center justify-between border-b border-slate-800 pb-6 mb-8">
        <div className="flex items-center gap-3">
          <ShieldAlert className="w-8 h-8 text-rose-500" />
          <div>
            <h1 className="text-xl font-bold tracking-tight">NTRO Network Attack Forecaster</h1>
            <p className="text-xs text-slate-400">SIH 26153 • State-Space Dynamics Engine</p>
          </div>
        </div>
        <div className="flex items-center gap-2 bg-slate-900 border border-slate-800 px-3 py-1.5 rounded-full text-xs text-emerald-400">
          <Radio className="w-4 h-4 animate-pulse" /> Live Telemetry Stream
        </div>
      </header>

      <main className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Controls Panel */}
        <section className="bg-slate-900/60 border border-slate-800 rounded-xl p-6">
          <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-400 mb-6 flex items-center gap-2">
            <Cpu className="w-4 h-4 text-cyan-400" /> Telemetry Perturbation
          </h2>

          <div className="space-y-6">
            <div>
              <div className="flex justify-between text-xs mb-2">
                <span>Recon / Port Scan Intensity</span>
                <span className="font-mono text-cyan-400">{(recon * 100).toFixed(0)}%</span>
              </div>
              <input type="range" min="0" max="1" step="0.05" value={recon} onChange={(e) => setRecon(parseFloat(e.target.value))} className="w-full accent-cyan-400" />
            </div>

            <div>
              <div className="flex justify-between text-xs mb-2">
                <span>Lateral Movement / TTL Variance</span>
                <span className="font-mono text-cyan-400">{(lateral * 100).toFixed(0)}%</span>
              </div>
              <input type="range" min="0" max="1" step="0.05" value={lateral} onChange={(e) => setLateral(parseFloat(e.target.value))} className="w-full accent-cyan-400" />
            </div>

            <div>
              <div className="flex justify-between text-xs mb-2">
                <span>Exfiltration Volume Surge</span>
                <span className="font-mono text-cyan-400">{(exfil * 100).toFixed(0)}%</span>
              </div>
              <input type="range" min="0" max="1" step="0.05" value={exfil} onChange={(e) => setExfil(parseFloat(e.target.value))} className="w-full accent-cyan-400" />
            </div>

            <button onClick={runSimulation} disabled={loading} className="w-full mt-4 flex items-center justify-center gap-2 bg-cyan-600 hover:bg-cyan-500 py-2.5 rounded-lg text-sm font-medium transition">
              <Play className="w-4 h-4" /> {loading ? "Simulating Rollout..." : "Run K-Step Simulation"}
            </button>
          </div>
        </section>

        {/* Dynamic Forecast View */}
        <section className="lg:col-span-2 space-y-6">
          <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-6">
            <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-400 mb-4 flex items-center gap-2">
              <Activity className="w-4 h-4 text-rose-400" /> Infiltration Risk Trajectory P(S_t+k | S_t)
            </h2>
            <div className="h-64 w-full">
              {results && (
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={results.timeline}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                    <XAxis dataKey="step" stroke="#64748b" fontSize={12} />
                    <YAxis stroke="#64748b" fontSize={12} domain={[0, 100]} />
                    <Tooltip contentStyle={{ backgroundColor: "#0f172a", borderColor: "#334155" }} />
                    <Line type="monotone" dataKey="risk_percentage" stroke="#f43f5e" strokeWidth={3} dot={{ r: 5 }} />
                  </LineChart>
                </ResponsiveContainer>
              )}
            </div>
          </div>

          {/* Stage Progression Cards */}
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
            {results?.timeline.map((step, idx) => (
              <div key={idx} className="bg-slate-900/40 border border-slate-800/80 p-3 rounded-lg">
                <span className="text-[10px] text-slate-500 uppercase">{step.step}</span>
                <p className="text-xs font-semibold text-rose-400 truncate">{step.predicted_stage}</p>
                <p className="text-xs text-slate-400 font-mono mt-1">Risk: {step.risk_percentage}%</p>
              </div>
            ))}
          </div>
        </section>
      </main>
    </div>
  );
}