// PEARLS AQI Predictor - Live Dashboard
// All data is fetched dynamically from the FastAPI inference backend.
// No hardcoded values, no dummy data, no static fallbacks.

import React, { useState, useEffect } from 'react';
import { 
  Activity, ShieldAlert, RefreshCw, MapPin, CheckCircle2, 
  Layers, Cpu, BarChart3, Globe, Database, Settings, LayoutDashboard, Search, Bell
} from 'lucide-react';
import { ResponsiveContainer, AreaChart, Area, XAxis, YAxis, Tooltip, CartesianGrid } from 'recharts';

export default function App() {
  const [activeNav, setActiveNav] = useState('dashboard');
  const [isLoading, setIsLoading] = useState(true);

  // Dynamic state â€” zero hardcoded city names or student values
  const [telemetry, setTelemetry] = useState({
    city: "",
    stationName: "",
    currentAQI: null,
    aqiStatus: "",
    aqiColor: "#64748B",
    aqiDelta: "",
    pm25: null,
    whoStatus: "",
    healthAdvisory: "",
    healthDetail: "",
    confidenceScore: null,
    modelName: "",
    featureStoreStatus: "Disconnected",
    forecasts: [],     // Array of horizon objects: [{ horizon: '24H', aqi: 0, status: '', color: '', rmse: 0 }]
    trendHistory: [],  // Array of trend objects: [{ time: '', aqi: 0, pm25: 0 }]
    hotspots: [],      // Array of regional nodes: [{ id: 1, name: '', aqi: 0, status: '', color: '', textColor: '' }]
    systemMetrics: {
      completeness: "--",
      accuracy: "--",
      status: "Idle"
    }
  });

  // Fetch telemetry from your FastAPI/Python backend endpoint
  const fetchTelemetryData = async () => {
    setIsLoading(true);
    const endpoints = [
      '/api/telemetry',
      'http://127.0.0.1:8000/api/telemetry',
      'http://localhost:8000/api/telemetry'
    ];
    let success = false;
    for (const endpoint of endpoints) {
      try {
        const response = await fetch(endpoint);
        if (response.ok) {
          const data = await response.json();
          setTelemetry(data);
          success = true;
          break;
        }
      } catch (error) {
        // Try next fallback endpoint
      }
    }
    if (!success) {
      console.error("Failed to connect to AQI backend endpoint across all configured URLs.");
    }
    setIsLoading(false);
  };

  useEffect(() => {
    fetchTelemetryData();
  }, []);

  return (
    <div style={{ display: 'flex', minHeight: '100vh', backgroundColor: '#F8FAFC', color: '#0F172A', fontFamily: 'system-ui, -apple-system, sans-serif' }}>
      
      {/* Sidebar */}
      <aside style={{ width: '260px', backgroundColor: '#0B132B', color: '#F8FAFC', display: 'flex', flexDirection: 'column', justifyContent: 'space-between', padding: '1.5rem', borderRight: '1px solid #1E293B', position: 'fixed', height: '100vh', boxSizing: 'border-box' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '2rem', paddingBottom: '1rem', borderBottom: '1px solid #1E293B' }}>
            <div style={{ backgroundColor: 'rgba(56, 189, 248, 0.15)', padding: '0.5rem', borderRadius: '0.5rem', border: '1px solid rgba(56, 189, 248, 0.3)' }}>
              <Activity size={22} color="#38BDF8" />
            </div>
            <div>
              <h2 style={{ fontSize: '1rem', fontWeight: '800', margin: 0, letterSpacing: '0.05em' }}>PEARLS AQI</h2>
              <span style={{ fontSize: '0.65rem', color: '#38BDF8', fontWeight: '600', letterSpacing: '0.08em' }}>MLOPS PLATFORM</span>
            </div>
          </div>

          <nav style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
            {['dashboard', 'regional', 'models', 'sources', 'settings'].map((navItem) => (
              <button 
                key={navItem} 
                onClick={() => setActiveNav(navItem)} 
                style={{ 
                  display: 'flex', alignItems: 'center', gap: '0.75rem', width: '100%', padding: '0.75rem 1rem', 
                  borderRadius: '0.5rem', 
                  backgroundColor: activeNav === navItem ? '#0284C7' : 'transparent', 
                  color: activeNav === navItem ? '#FFFFFF' : '#94A3B8', 
                  border: 'none', cursor: 'pointer', fontWeight: '600', fontSize: '0.85rem', textTransform: 'capitalize' 
                }}
              >
                {navItem === 'dashboard' && <LayoutDashboard size={16} />}
                {navItem === 'regional' && <Globe size={16} />}
                {navItem === 'models' && <Cpu size={16} />}
                {navItem === 'sources' && <Database size={16} />}
                {navItem === 'settings' && <Settings size={16} />}
                {navItem.replace('-', ' ')}
              </button>
            ))}
          </nav>
        </div>

        <div style={{ backgroundColor: '#1C2541', padding: '0.75rem 1rem', borderRadius: '0.75rem', border: '1px solid #334155', display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <div style={{ backgroundColor: '#0284C7', width: '32px', height: '32px', borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 'bold', fontSize: '0.8rem', color: '#FFFFFF' }}>MA</div>
          <div>
            <div style={{ fontSize: '0.8rem', fontWeight: '700', color: '#FFFFFF' }}>Mohsin Ali</div>
            <div style={{ fontSize: '0.65rem', color: '#34D399' }}>â— System Active</div>
          </div>
        </div>
      </aside>

      {/* Main Workspace */}
      <main style={{ marginLeft: '260px', flex: 1, padding: '2rem', display: 'flex', flexDirection: 'column', gap: '1.5rem', boxSizing: 'border-box' }}>
        
        {/* Header Controls */}
        <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', backgroundColor: '#FFFFFF', padding: '0.85rem 1.5rem', borderRadius: '1rem', border: '1px solid #E2E8F0' }}>
          <div style={{ fontSize: '0.9rem', fontWeight: '600', color: '#64748B' }}>
            Pearls Air Quality Intelligence Control Center
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', width: '35%' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', backgroundColor: '#F8FAFC', padding: '0.4rem 0.75rem', borderRadius: '0.5rem', border: '1px solid #CBD5E1', width: '100%', fontSize: '0.8rem', color: '#64748B' }}>
              <Search size={14} />
              <span>Search metrics, features, or regions...</span>
            </div>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
            <Bell size={18} color="#64748B" style={{ cursor: 'pointer' }} />
            <button onClick={fetchTelemetryData} style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', backgroundColor: '#0284C7', color: 'white', padding: '0.4rem 0.8rem', borderRadius: '0.5rem', border: 'none', cursor: 'pointer', fontSize: '0.8rem', fontWeight: '600' }}>
              <RefreshCw size={13} className={isLoading ? 'animate-spin' : ''} />
              <span>Sync Telemetry</span>
            </button>
          </div>
        </header>

        {/* Dynamic Location Title Bar */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <h2 style={{ fontSize: '1.35rem', fontWeight: '800', margin: 0, color: '#0F172A' }}>Dashboard Overview</h2>
          
          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
            <div style={{ backgroundColor: '#FFFFFF', border: '1px solid #CBD5E1', padding: '0.4rem 0.8rem', borderRadius: '0.5rem', fontSize: '0.8rem', fontWeight: '500', color: '#334155', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
              <MapPin size={14} color="#0284C7" />
              <span>{telemetry.city ? `${telemetry.city} ${telemetry.stationName ? `(${telemetry.stationName})` : ''}` : "Location Unset"}</span>
            </div>
            <div style={{ backgroundColor: telemetry.featureStoreStatus === "Connected" ? '#DCFCE7' : '#FEF3C7', border: `1px solid ${telemetry.featureStoreStatus === "Connected" ? '#86EFAC' : '#FCD34D'}`, padding: '0.4rem 0.8rem', borderRadius: '0.5rem', fontSize: '0.75rem', fontWeight: '700', color: telemetry.featureStoreStatus === "Connected" ? '#166534' : '#92400E', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
              <span style={{ width: '8px', height: '8px', backgroundColor: telemetry.featureStoreStatus === "Connected" ? '#22C55E' : '#F59E0B', borderRadius: '50%', display: 'inline-block' }}></span>
              {telemetry.featureStoreStatus}
            </div>
          </div>
        </div>

        {/* Metric Cards Grid */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1.25rem' }}>
          
          {/* Card 1: AQI Score */}
          <div style={{ backgroundColor: '#FFFFFF', padding: '1.25rem', borderRadius: '1rem', border: '1px solid #E2E8F0', display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
            <span style={{ fontSize: '0.8rem', fontWeight: '700', color: '#0F172A' }}>Current AQI</span>
            <div style={{ textAlign: 'center', padding: '0.5rem 0' }}>
              <div style={{ fontSize: '2.5rem', fontWeight: '900', color: telemetry.aqiColor, lineHeight: '1' }}>
                {telemetry.currentAQI !== null ? telemetry.currentAQI : "--"}
              </div>
              <div style={{ fontSize: '0.75rem', fontWeight: '700', color: telemetry.aqiColor, marginTop: '0.3rem' }}>
                {telemetry.aqiStatus || "Awaiting Data"}
              </div>
            </div>
            <span style={{ fontSize: '0.65rem', color: '#64748B', textAlign: 'center' }}>{telemetry.aqiDelta}</span>
          </div>

          {/* Card 2: Main Pollutant */}
          <div style={{ backgroundColor: '#FFFFFF', padding: '1.25rem', borderRadius: '1rem', border: '1px solid #E2E8F0', display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
            <div>
              <span style={{ fontSize: '0.8rem', fontWeight: '700', color: '#0F172A' }}>Main Pollutant</span>
              <div style={{ fontSize: '0.85rem', fontWeight: '700', color: '#0F172A', marginTop: '0.4rem' }}>PM2.5 Concentration</div>
              <div style={{ fontSize: '1.25rem', fontWeight: '800', color: '#0284C7', marginTop: '0.2rem' }}>
                {telemetry.pm25 !== null ? `${telemetry.pm25} Âµg/mÂ³` : "--"}
              </div>
            </div>
            <div style={{ fontSize: '0.7rem', color: '#166534', fontWeight: '600' }}>{telemetry.whoStatus}</div>
          </div>

          {/* Card 3: Health Advisory */}
          <div style={{ backgroundColor: '#FFFFFF', padding: '1.25rem', borderRadius: '1rem', border: '1px solid #E2E8F0', display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
            <div>
              <span style={{ fontSize: '0.8rem', fontWeight: '700', color: '#0F172A' }}>Health Advisory</span>
              <div style={{ fontSize: '0.8rem', fontWeight: '700', color: '#D97706', marginTop: '0.4rem' }}>
                {telemetry.healthAdvisory || "No Advisory"}
              </div>
            </div>
            <div style={{ fontSize: '0.7rem', color: '#64748B', marginTop: '0.5rem' }}>{telemetry.healthDetail}</div>
          </div>

          {/* Card 4: Confidence Score */}
          <div style={{ backgroundColor: '#FFFFFF', padding: '1.25rem', borderRadius: '1rem', border: '1px solid #E2E8F0', display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
            <div>
              <span style={{ fontSize: '0.8rem', fontWeight: '700', color: '#0F172A' }}>Forecast Confidence</span>
              <div style={{ fontSize: '2.2rem', fontWeight: '900', color: '#0284C7', marginTop: '0.2rem' }}>
                {telemetry.confidenceScore !== null ? `${telemetry.confidenceScore}%` : "--"}
              </div>
            </div>
            <div style={{ fontSize: '0.7rem', color: '#64748B' }}>{telemetry.modelName ? `Active Model: ${telemetry.modelName}` : "Model Unlinked"}</div>
          </div>

        </div>

        {/* Multi-Horizon Projections & Dynamic Chart */}
        <div style={{ backgroundColor: '#FFFFFF', padding: '1.5rem', borderRadius: '1rem', border: '1px solid #E2E8F0' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
            <h3 style={{ fontSize: '0.95rem', fontWeight: '800', color: '#0F172A', margin: 0, display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <Layers size={18} color="#0284C7" /> Multi-Horizon Projections & Forecast Trend
            </h3>
            <span style={{ fontSize: '0.75rem', color: '#64748B', fontWeight: '600' }}>
              Feature Store Pipeline: {telemetry.featureStoreStatus}
            </span>
          </div>

          {/* Forecast Cards dynamically mapped from backend array */}
          <div style={{ display: 'grid', gridTemplateColumns: telemetry.forecasts.length > 0 ? `repeat(${telemetry.forecasts.length}, 1fr)` : '1fr', gap: '1rem', marginBottom: '1.25rem' }}>
            {telemetry.forecasts.length > 0 ? (
              telemetry.forecasts.map((item, idx) => (
                <div key={idx} style={{ backgroundColor: '#F8FAFC', padding: '1rem', borderRadius: '0.75rem', border: '1px solid #E2E8F0' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', fontWeight: '700' }}>
                    <span style={{ color: '#0284C7' }}>{item.horizon} HORIZON</span>
                    <span style={{ color: item.color || '#64748B' }}>{item.status}</span>
                  </div>
                  <div style={{ fontSize: '1.75rem', fontWeight: '900', color: '#0F172A', margin: '0.2rem 0' }}>
                    {item.aqi} <span style={{ fontSize: '0.75rem', fontWeight: 'normal', color: '#64748B' }}>AQI</span>
                  </div>
                  <div style={{ fontSize: '0.7rem', color: '#64748B' }}>Validation RMSE: Â±{item.rmse}</div>
                </div>
              ))
            ) : (
              <div style={{ padding: '1rem', textAlign: 'center', color: '#94A3B8', fontSize: '0.85rem' }}>
                No multi-horizon forecasts loaded from backend.
              </div>
            )}
          </div>

          {/* Recharts Area Chart dynamically mapped from backend trendHistory */}
          <div style={{ height: '200px', width: '100%' }}>
            {telemetry.trendHistory.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={telemetry.trendHistory}>
                  <defs>
                    <linearGradient id="colorAqi" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#0284C7" stopOpacity={0.3}/>
                      <stop offset="95%" stopColor="#0284C7" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                  <XAxis dataKey="time" stroke="#64748B" fontSize={11} />
                  <YAxis stroke="#64748B" fontSize={11} />
                  <Tooltip contentStyle={{ backgroundColor: '#0F172A', borderColor: '#334155', borderRadius: '8px', color: '#fff', fontSize: '12px' }} />
                  <Area type="monotone" dataKey="aqi" stroke="#0284C7" strokeWidth={2.5} fillOpacity={1} fill="url(#colorAqi)" />
                </AreaChart>
              </ResponsiveContainer>
            ) : (
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: '#94A3B8', fontSize: '0.85rem', border: '1px dashed #CBD5E1', borderRadius: '0.5rem' }}>
                Waiting for trend data array from server...
              </div>
            )}
          </div>
        </div>

        {/* Regional Hotspots & System Metrics */}
        <div style={{ display: 'grid', gridTemplateColumns: '1.4fr 1fr', gap: '1.25rem' }}>
          
          <div style={{ backgroundColor: '#FFFFFF', padding: '1.5rem', borderRadius: '1rem', border: '1px solid #E2E8F0' }}>
            <h3 style={{ fontSize: '0.9rem', fontWeight: '800', color: '#0F172A', margin: '0 0 1rem 0' }}>Regional Hotspots</h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
              {telemetry.hotspots.length > 0 ? (
                telemetry.hotspots.map((spot) => (
                  <div key={spot.id} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '0.5rem 0', borderBottom: '1px solid #F1F5F9' }}>
                    <span style={{ fontSize: '0.85rem', fontWeight: '600', color: '#334155' }}>{spot.name}</span>
                    <span style={{ fontSize: '0.8rem', fontWeight: '700', backgroundColor: spot.color || '#F1F5F9', color: spot.textColor || '#0F172A', padding: '0.2rem 0.6rem', borderRadius: '4px' }}>
                      {spot.aqi}
                    </span>
                  </div>
                ))
              ) : (
                <span style={{ fontSize: '0.8rem', color: '#94A3B8' }}>No regional hotspot stations mapped.</span>
              )}
            </div>
          </div>

          <div style={{ backgroundColor: '#FFFFFF', padding: '1.5rem', borderRadius: '1rem', border: '1px solid #E2E8F0', display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
            <h3 style={{ fontSize: '0.9rem', fontWeight: '800', color: '#0F172A', margin: '0 0 1rem 0' }}>Data Integrity & Pipeline Metrics</h3>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', textAlign: 'center' }}>
              <div style={{ backgroundColor: '#F8FAFC', padding: '0.75rem', borderRadius: '0.5rem', border: '1px solid #E2E8F0' }}>
                <div style={{ fontSize: '1.25rem', fontWeight: '900', color: '#0284C7' }}>{telemetry.systemMetrics.completeness}</div>
                <div style={{ fontSize: '0.7rem', color: '#64748B', fontWeight: '600' }}>Completeness</div>
              </div>
              <div style={{ backgroundColor: '#F8FAFC', padding: '0.75rem', borderRadius: '0.5rem', border: '1px solid #E2E8F0' }}>
                <div style={{ fontSize: '1.25rem', fontWeight: '900', color: '#10B981' }}>{telemetry.systemMetrics.accuracy}</div>
                <div style={{ fontSize: '0.7rem', color: '#64748B', fontWeight: '600' }}>Sensor Accuracy</div>
              </div>
            </div>
            <div style={{ backgroundColor: '#F8FAFC', border: '1px solid #E2E8F0', padding: '0.6rem', borderRadius: '0.5rem', textAlign: 'center', marginTop: '1rem' }}>
              <span style={{ fontSize: '0.75rem', fontWeight: '700', color: '#334155' }}>{telemetry.systemMetrics.status}</span>
            </div>
          </div>

        </div>

      </main>

    </div>
  );
}
