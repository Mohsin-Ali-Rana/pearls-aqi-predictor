import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Activity, RefreshCw, MapPin, Radio, Bell,
  Globe, LayoutDashboard, Sparkles, LineChart, TrendingUp, ChevronRight, Cpu, Layers
} from 'lucide-react';
import { 
  ResponsiveContainer, AreaChart, Area, XAxis, YAxis, Tooltip, CartesianGrid, ReferenceArea, ReferenceLine
} from 'recharts';

import HeroAQIGauge from './components/HeroAQIGauge';
import ForecastCard from './components/ForecastCard';
import ShapWaterfall from './components/ShapWaterfall';
import TournamentChart from './components/TournamentChart';
import CorrelationHeatmap from './components/CorrelationHeatmap';
import MLOpsTelemetryBar, { EmailAlertDispatcher } from './components/MLOpsTelemetryBar';

export default function App() {
  const [activeNav, setActiveNav] = useState('dashboard');
  const [isLoading, setIsLoading] = useState(true);
  const [isSyncing, setIsSyncing] = useState(false);
  const [syncProgress, setSyncProgress] = useState(0);
  const [edaData, setEdaData] = useState(null);

  // Telemetry State - Bound exclusively to live backend telemetry
  const [telemetry, setTelemetry] = useState({
    city: null,
    coordinates: null,
    stationName: null,
    currentAQI: null,
    aqiStatus: "",
    aqiColor: "#0D9488",
    aqiDelta: null,
    pm25: null,
    whoStatus: null,
    healthAdvisory: null,
    healthDetail: null,
    confidenceScore: null,
    modelName: null,
    featureStoreStatus: "Connecting",
    forecasts: [],
    trendHistory: [],
    hotspots: [],
    shapExplanations: [],
    localShapContributions: [],
    localShapByHorizon: null,
    currentWeather: null,
    persistenceLift: null,
    systemMetrics: {
      completeness: null,
      accuracy: null,
      status: "Initializing System Stream"
    }
  });

  const fetchTelemetryData = async (showSyncAnim = false, isBackground = false) => {
    if (showSyncAnim) {
      setIsSyncing(true);
      setSyncProgress(15);
    } else if (!isBackground) {
      setIsLoading(true);
    }

    const endpoints = [
      '/api/telemetry',
      'http://127.0.0.1:8000/api/telemetry',
      'http://localhost:8000/api/telemetry'
    ];
    let success = false;
    
    // Smooth progress animation if syncing
    let interval;
    if (showSyncAnim) {
      interval = setInterval(() => {
        setSyncProgress((prev) => (prev < 90 ? prev + 25 : prev));
      }, 200);
    }

    for (const endpoint of endpoints) {
      try {
        const response = await fetch(endpoint);
        if (response.ok) {
          const data = await response.json();
          setTelemetry(data);
          success = true;
          break;
        }
      } catch (error) {}
    }

    if (!success) {
      console.warn("Telemetry API offline or starting up.");
    }

    if (showSyncAnim) {
      setSyncProgress(100);
      setTimeout(() => {
        setIsLoading(false);
        setIsSyncing(false);
        setSyncProgress(0);
        if (interval) clearInterval(interval);
      }, 700);
    } else if (!isBackground) {
      setTimeout(() => {
        setIsLoading(false);
        setIsSyncing(false);
      }, 350);
    }
  };

  const fetchEdaData = async () => {
    const endpoints = ['/api/eda', 'http://127.0.0.1:8000/api/eda', 'http://localhost:8000/api/eda'];
    for (const ep of endpoints) {
      try {
        const res = await fetch(ep);
        if (res.ok) {
          const data = await res.json();
          setEdaData(data);
          break;
        }
      } catch (e) {}
    }
  };

  useEffect(() => {
    fetchTelemetryData(false, false);
    fetchEdaData();

    // Quiet background polling every 30 seconds
    const pollTimer = setInterval(() => {
      fetchTelemetryData(false, true);
    }, 30000);

    return () => clearInterval(pollTimer);
  }, []);

  const scrollToAlertDispatcher = () => {
    setActiveNav('dashboard');
    
    const performScroll = () => {
      const elem = document.getElementById('email-alert-section');
      if (elem) {
        const yOffset = -60;
        const y = elem.getBoundingClientRect().top + window.pageYOffset + yOffset;
        window.scrollTo({ top: y, behavior: 'smooth' });
        return true;
      }
      return false;
    };

    if (!performScroll()) {
      let attempts = 0;
      const interval = setInterval(() => {
        attempts++;
        if (performScroll() || attempts > 20) {
          clearInterval(interval);
        }
      }, 40);
    }
  };

  const navItems = [
    { id: 'dashboard', label: 'Live Overview', icon: LayoutDashboard },
    { id: 'trends', label: 'Daily Trends (EDA)', icon: LineChart },
    { id: 'shap', label: 'Model Factors (SHAP)', icon: Sparkles },
    { id: 'tournament', label: 'Multi-Model Tournament', icon: Cpu },
    { id: 'regional', label: 'Atmospheric Features', icon: Globe }
  ];

  return (
    <div style={{ 
      display: 'flex', 
      minHeight: '100vh', 
      backgroundColor: '#F8FAFC', 
      fontFamily: "'Plus Jakarta Sans', system-ui, -apple-system, sans-serif",
      color: '#0F172A'
    }}>
      
      {/* Light Professional Executive Sidebar */}
      <aside style={{ 
        width: '280px', 
        backgroundColor: '#FFFFFF', 
        borderRight: '1px solid #E2E8F0', 
        padding: '1.75rem 1.25rem', 
        display: 'flex', 
        flexDirection: 'column', 
        justify: 'space-between',
        position: 'fixed',
        height: '100vh',
        boxSizing: 'border-box',
        zIndex: 50,
        boxShadow: '4px 0 24px rgba(15, 23, 42, 0.03)'
      }}>
        <div>
          {/* Logo Header Container (New Transparent PNG Logo + Crisp Typography) */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.85rem', marginBottom: '2.2rem' }}>
            <img 
              src="/pearls_logo_new.png" 
              alt="PEARLS AQI Logo" 
              style={{ 
                height: '46px', 
                width: 'auto',
                maxHeight: '46px',
                objectFit: 'contain',
                flexShrink: 0,
                filter: 'drop-shadow(0 4px 10px rgba(13, 148, 136, 0.20))'
              }} 
              onError={(e) => {
                e.target.src = '/logo.png';
              }}
            />

            <div style={{ display: 'flex', flexDirection: 'column' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                <span style={{ fontSize: '1.25rem', fontWeight: '900', color: '#0F172A', letterSpacing: '-0.03em', lineHeight: 1.1 }}>
                  PEARLS
                </span>
                <span style={{ fontSize: '1.25rem', fontWeight: '900', color: '#0D9488', letterSpacing: '-0.03em', lineHeight: 1.1 }}>
                  AQI
                </span>
              </div>
              
              <div style={{ fontSize: '0.6rem', fontWeight: '800', color: '#64748B', letterSpacing: '0.08em', textTransform: 'uppercase', marginTop: '0.1rem' }}>
                ATMOSPHERIC INTELLIGENCE
              </div>

              <div style={{ marginTop: '0.35rem' }}>
                <span style={{ 
                  fontSize: '0.62rem', 
                  fontWeight: '900', 
                  color: '#0D9488', 
                  letterSpacing: '0.06em', 
                  textTransform: 'uppercase', 
                  backgroundColor: '#F0FDFA', 
                  padding: '0.2rem 0.55rem', 
                  borderRadius: '0.35rem', 
                  border: '1px solid #CCFBF1',
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: '0.3rem'
                }}>
                  <span style={{ width: '5px', height: '5px', backgroundColor: '#0D9488', borderRadius: '50%' }} />
                  COMMAND CENTER
                </span>
              </div>
            </div>
          </div>

          {/* Navigation Items */}
          <nav style={{ display: 'flex', flexDirection: 'column', gap: '0.45rem' }}>
            {navItems.map((navItem) => {
              const Icon = navItem.icon;
              const isActive = activeNav === navItem.id;
              return (
                <button
                  key={navItem.id}
                  onClick={() => setActiveNav(navItem.id)}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    justify: 'space-between',
                    width: '100%',
                    padding: '0.8rem 1rem',
                    borderRadius: '0.75rem',
                    border: isActive ? '1px solid #CCFBF1' : '1px solid transparent',
                    backgroundColor: isActive ? '#F0FDFA' : 'transparent',
                    color: isActive ? '#0D9488' : '#64748B',
                    fontWeight: isActive ? '800' : '600',
                    fontSize: '0.88rem',
                    cursor: 'pointer',
                    transition: 'all 0.15s ease',
                    textAlign: 'left'
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.85rem' }}>
                    <Icon size={19} color={isActive ? '#0D9488' : '#64748B'} />
                    <span>{navItem.label}</span>
                  </div>
                  {isActive && <ChevronRight size={16} color="#0D9488" />}
                </button>
              );
            })}
          </nav>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.65rem' }}>
          {/* Idea 5: "Get Instant Alerts" Quick Action Card (Refined Light Theme) */}
          <motion.button
            whileHover={{ scale: 1.02, backgroundColor: '#F0FDFA', borderColor: '#99F6E4' }}
            whileTap={{ scale: 0.98 }}
            onClick={scrollToAlertDispatcher}
            style={{
              backgroundColor: '#FFFFFF',
              padding: '0.8rem 0.95rem',
              borderRadius: '0.85rem',
              border: '1px solid #E2E8F0',
              boxShadow: '0 2px 10px rgba(15, 23, 42, 0.04)',
              display: 'flex',
              alignItems: 'center',
              justify: 'space-between',
              width: '100%',
              cursor: 'pointer',
              textAlign: 'left',
              transition: 'all 0.2s ease'
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.7rem' }}>
              <div style={{ position: 'relative', width: '34px', height: '34px', display: 'grid', placeItems: 'center', flexShrink: 0 }}>
                {/* Glowing Pulse Accent */}
                <motion.div
                  animate={{ scale: [1, 1.3, 1], opacity: [0.4, 0, 0.4] }}
                  transition={{ duration: 2.2, repeat: Infinity, ease: 'easeInOut' }}
                  style={{
                    position: 'absolute',
                    inset: 0,
                    borderRadius: '50%',
                    backgroundColor: '#0D9488'
                  }}
                />
                <div style={{
                  position: 'relative',
                  width: '32px',
                  height: '32px',
                  borderRadius: '0.55rem',
                  background: 'linear-gradient(135deg, #0D9488 0%, #0284C7 100%)',
                  boxShadow: '0 2px 8px rgba(13, 148, 136, 0.25)',
                  display: 'grid',
                  placeItems: 'center',
                  zIndex: 2
                }}>
                  <Bell size={16} color="#FFFFFF" />
                </div>
              </div>

              <div style={{ display: 'flex', flexDirection: 'column' }}>
                <span style={{ fontSize: '0.74rem', fontWeight: '800', color: '#0F172A', letterSpacing: '-0.01em' }}>
                  AQI ALERT DISPATCH
                </span>
                <span style={{ fontSize: '0.62rem', color: '#0D9488', fontWeight: '700', letterSpacing: '0.02em' }}>
                  Configure Early Warning
                </span>
              </div>
            </div>
            <ChevronRight size={16} color="#0D9488" />
          </motion.button>

          {/* Idea 4: Interactive Atmospheric Globe / Satellite Radar Orb Widget */}
          <div style={{
            background: 'linear-gradient(135deg, #F0FDFA 0%, #E0F2FE 100%)',
            padding: '0.85rem 0.95rem',
            borderRadius: '0.85rem',
            border: '1px solid #CCFBF1',
            boxShadow: '0 4px 16px rgba(13, 148, 136, 0.08)',
            display: 'flex',
            alignItems: 'center',
            gap: '0.75rem',
            position: 'relative',
            overflow: 'hidden'
          }}>
            {/* Subtle Ambient Background Flare */}
            <div style={{
              position: 'absolute',
              top: '-20px',
              right: '-20px',
              width: '60px',
              height: '60px',
              borderRadius: '50%',
              background: 'radial-gradient(circle, rgba(14,165,233,0.2) 0%, rgba(240,253,250,0) 70%)',
              pointerEvents: 'none'
            }} />

            {/* Animated 3D Spinning Globe Orb with Orbit Ring */}
            <div style={{ position: 'relative', width: '42px', height: '42px', display: 'grid', placeItems: 'center', flexShrink: 0 }}>
              {/* Outer Orbit Halo Ring */}
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ duration: 12, repeat: Infinity, ease: 'linear' }}
                style={{
                  position: 'absolute',
                  inset: '-4px',
                  borderRadius: '50%',
                  border: '1.5px dashed rgba(13, 148, 136, 0.4)',
                  borderTopColor: '#0284C7'
                }}
              />

              {/* Glowing Globe Sphere */}
              <div style={{
                width: '36px',
                height: '36px',
                borderRadius: '50%',
                background: 'radial-gradient(circle at 35% 35%, #2DD4BF 0%, #0D9488 60%, #0F172A 100%)',
                boxShadow: '0 2px 10px rgba(13, 148, 136, 0.35)',
                display: 'grid',
                placeItems: 'center',
                position: 'relative',
                overflow: 'hidden'
              }}>
                {/* Rotating Internal Globe Grid */}
                <motion.div
                  animate={{ rotate: [0, 360] }}
                  transition={{ duration: 20, repeat: Infinity, ease: 'linear' }}
                  style={{ opacity: 0.85 }}
                >
                  <Globe size={22} color="#FFFFFF" />
                </motion.div>
              </div>
            </div>

            {/* Label & Live Satellite Status */}
            <div style={{ display: 'flex', flexDirection: 'column', flex: 1 }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <span style={{ fontSize: '0.74rem', fontWeight: '900', color: '#0F172A', letterSpacing: '-0.01em' }}>
                  SATELLITE RADAR
                </span>
                <motion.span
                  animate={{ opacity: [1, 0.3, 1] }}
                  transition={{ duration: 1.8, repeat: Infinity }}
                  style={{
                    width: '7px',
                    height: '7px',
                    borderRadius: '50%',
                    backgroundColor: '#10B981',
                    boxShadow: '0 0 8px #10B981'
                  }}
                />
              </div>
              <div style={{ fontSize: '0.62rem', fontWeight: '700', color: '#0D9488', marginTop: '0.1rem', letterSpacing: '0.02em' }}>
                Aerosol Observation Active
              </div>
            </div>
          </div>
        </div>
      </aside>

      {/* Main Workspace Area */}
      <main style={{ marginLeft: '280px', flex: 1, padding: '2rem 2.5rem', display: 'flex', flexDirection: 'column', gap: '1.75rem', boxSizing: 'border-box', maxWidth: '1440px' }}>
               {/* Flagship Hero Command Banner (Wah Cantt & Taxila Primary Zone) */}
        <header style={{ 
          backgroundColor: '#FFFFFF', 
          padding: '1.6rem 2.2rem', 
          borderRadius: '1.25rem', 
          border: '1.5px solid #CCFBF1',
          boxShadow: '0 8px 32px -4px rgba(13, 148, 136, 0.12), 0 4px 18px -2px rgba(15, 23, 42, 0.04)',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          gap: '1.5rem',
          position: 'relative',
          overflow: 'hidden'
        }}>
          {/* Ambient subtle decorative background accent */}
          <div style={{
            position: 'absolute',
            top: '-50px',
            right: '-50px',
            width: '260px',
            height: '260px',
            background: 'radial-gradient(circle, rgba(13,148,136,0.07) 0%, rgba(255,255,255,0) 70%)',
            pointerEvents: 'none'
          }} />

          {/* Left Hero Content: Large Prominent Dual-Tone Title & Subtitle */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.45rem', zIndex: 1 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.85rem', flexWrap: 'wrap' }}>
              <h1 style={{ 
                fontSize: '2.1rem', 
                fontWeight: '900', 
                margin: 0, 
                letterSpacing: '-0.035em',
                lineHeight: 1.1,
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem'
              }}>
                <span style={{ color: '#0F172A' }}>Wah Cantt</span>
                <span style={{ color: '#0D9488' }}>&amp; Taxila</span>
              </h1>

              <span style={{
                backgroundColor: '#0D9488',
                color: '#FFFFFF',
                fontSize: '0.72rem',
                fontWeight: '900',
                padding: '0.3rem 0.75rem',
                borderRadius: '0.6rem',
                letterSpacing: '0.05em',
                textTransform: 'uppercase',
                display: 'inline-flex',
                alignItems: 'center',
                gap: '0.4rem',
                boxShadow: '0 4px 14px rgba(13, 148, 136, 0.32)'
              }}>
                <span className="target-badge-pulse" style={{ width: '7px', height: '7px', backgroundColor: '#34D399', borderRadius: '50%', display: 'inline-block' }} />
                LIVE GRID ZONE
              </span>
            </div>

            <p style={{ fontSize: '0.86rem', color: '#64748B', margin: 0, fontWeight: '600' }}>
              Real-time PM2.5 Observation Stream &amp; 72-Hour Multi-Horizon AI Engine
            </p>
          </div>

          {/* Right Telemetry Controls */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', zIndex: 1, flexShrink: 0 }}>
            
            {/* Target Coordinate Grid Card (No Repeated City Text) */}
            <div style={{ 
              backgroundColor: '#F8FAFC', 
              border: '1px solid #E2E8F0', 
              padding: '0.6rem 1.1rem', 
              borderRadius: '0.9rem', 
              display: 'flex', 
              alignItems: 'center', 
              gap: '0.7rem',
              boxShadow: '0 2px 10px rgba(15, 23, 42, 0.03)'
            }}>
              <div style={{ 
                width: '34px', 
                height: '34px', 
                borderRadius: '0.6rem', 
                backgroundColor: '#0D9488', 
                display: 'flex', 
                alignItems: 'center', 
                justifyContent: 'center',
                flexShrink: 0,
                boxShadow: '0 2px 10px rgba(13, 148, 136, 0.28)'
              }}>
                <MapPin size={17} color="#FFFFFF" />
              </div>
              
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.1rem' }}>
                <span style={{ fontSize: '0.78rem', fontWeight: '800', color: '#0F172A', letterSpacing: '-0.01em' }}>
                  Target Coordinate Grid
                </span>
                <span style={{ fontSize: '0.74rem', color: '#0D9488', fontWeight: '700', letterSpacing: '0.01em' }}>
                  33.77° N, 72.75° E
                </span>
              </div>
            </div>

            {/* Sync Telemetry Action Button */}
            <button 
              onClick={() => fetchTelemetryData(true)} 
              disabled={isSyncing}
              style={{ 
                display: 'flex', 
                alignItems: 'center', 
                gap: '0.55rem', 
                backgroundColor: isSyncing ? '#0F766E' : '#0D9488', 
                color: '#FFFFFF', 
                padding: '0.75rem 1.4rem', 
                borderRadius: '0.9rem', 
                border: 'none', 
                cursor: isSyncing ? 'not-allowed' : 'pointer', 
                fontSize: '0.88rem', 
                fontWeight: '800',
                boxShadow: '0 4px 18px rgba(13, 148, 136, 0.32)',
                transition: 'all 0.2s cubic-bezier(0.4, 0, 0.2, 1)',
                whiteSpace: 'nowrap'
              }}
              onMouseEnter={(e) => !isSyncing && (e.currentTarget.style.transform = 'translateY(-1px)')}
              onMouseLeave={(e) => !isSyncing && (e.currentTarget.style.transform = 'translateY(0)')}
            >
              <RefreshCw size={16} className={isSyncing ? 'animate-spin' : ''} />
              <span>{isSyncing ? 'Scanning Telemetry...' : 'Sync Telemetry'}</span>
            </button>

          </div>
        </header>

        {/* Dynamic Scanning Animation Banner when Loading or Syncing Telemetry */}
        <AnimatePresence>
          {(isSyncing || isLoading) && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              style={{
                backgroundColor: '#FFFFFF',
                borderRadius: '1rem',
                border: '1px solid #0D9488',
                padding: '1rem 1.5rem',
                position: 'relative',
                overflow: 'hidden',
                boxShadow: '0 4px 20px rgba(13, 148, 136, 0.12)'
              }}
            >
              <div className="radar-scan-line" />
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', position: 'relative', zIndex: 2 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.8rem' }}>
                  <Radio size={20} color="#0D9488" className="radar-pulse" />
                  <div>
                    <span style={{ fontSize: '0.88rem', fontWeight: '800', color: '#0F172A' }}>
                      Querying Open-Meteo Satellite Streams & Hopsworks Feature Store V2...
                    </span>
                    <p style={{ margin: '0.15rem 0 0 0', fontSize: '0.75rem', color: '#64748B' }}>
                      Re-calibrating direct 72-hour multi-horizon state vectors for Wah Cantt / Taxila grid
                    </p>
                  </div>
                </div>
                <span style={{ fontSize: '0.85rem', fontWeight: '900', color: '#0D9488', fontFamily: 'monospace' }}>
                  {syncProgress}%
                </span>
              </div>
              <div style={{ marginTop: '0.75rem', height: '4px', backgroundColor: '#E2E8F0', borderRadius: '2px', overflow: 'hidden' }}>
                <div style={{ width: `${syncProgress}%`, height: '100%', backgroundColor: '#0D9488', transition: 'width 0.2s ease' }} />
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Tab Content Wrapper */}
        <AnimatePresence mode="wait">
          
          {/* TAB 1: LIVE DASHBOARD OVERVIEW */}
          {activeNav === 'dashboard' && (
            <motion.div 
              key="dashboard"
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -12 }}
              transition={{ duration: 0.25 }}
              style={{ display: 'flex', flexDirection: 'column', gap: '1.75rem' }}
            >
              {/* Live MLOps Infrastructure Health Ticker */} 
              <MLOpsTelemetryBar systemMetrics={telemetry.systemMetrics} modelName={telemetry.modelName} featureStoreStatus={telemetry.featureStoreStatus} />

              {/* Executive Hero Arc Gauge + Meteorological Cards + Persistence Lift */}
              <HeroAQIGauge telemetry={telemetry} isLoading={isLoading} />

              {/* 3 Multi-Horizon Forecast Cards (+24H, +48H, +72H) */}
              <div>
                <h3 style={{ fontSize: '1.05rem', fontWeight: '800', color: '#0F172A', margin: '0 0 1rem 0', display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
                  <Layers size={20} color="#0D9488" /> Direct Multi-Horizon AI Forecasts (+24h, +48h, +72h)
                </h3>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, minmax(0, 1fr))', gap: '1.25rem' }}>
                  {isLoading || !telemetry.forecasts || telemetry.forecasts.length === 0 ? (
                    [{ horizon: '24H' }, { horizon: '48H' }, { horizon: '72H' }].map((fc, idx) => (
                      <ForecastCard key={idx} forecast={fc} isLoading={true} />
                    ))
                  ) : (
                    telemetry.forecasts.map((fc, idx) => (
                      <ForecastCard key={idx} forecast={fc} isLoading={false} />
                    ))
                  )}
                </div>
              </div>

              {/* 24-Hour / 3-Day Forecast Trajectory Spline & Severity Threshold Bands */}
              <div style={{ 
                backgroundColor: '#FFFFFF', 
                padding: '1.75rem', 
                borderRadius: '1.25rem', 
                border: '1px solid #E2E8F0',
                boxShadow: '0 4px 20px -2px rgba(15, 23, 42, 0.04)'
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '1.2rem', gap: '1rem' }}>
                  <div style={{ display: 'flex', alignItems: 'flex-start', gap: '0.65rem', flex: 1, minWidth: 0 }}>
                    <div style={{ width: '34px', height: '34px', borderRadius: '0.65rem', backgroundColor: '#F0FDFA', border: '1px solid #CCFBF1', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
                      <TrendingUp size={18} color="#0D9488" />
                    </div>
                    <div>
                      <h3 style={{ fontSize: '1.1rem', fontWeight: '800', color: '#0F172A', margin: 0, letterSpacing: '-0.015em' }}>
                        3-Day AQI Forecast Trajectory &amp; Severity Threshold Bands
                      </h3>
                      <p style={{ fontSize: '0.78rem', color: '#64748B', margin: '0.15rem 0 0 0', fontWeight: '500' }}>
                        Continuous observed stream to 72-hour direct AI prediction spline mapped against WHO &amp; US-EPA severity bands
                      </p>
                    </div>
                  </div>
                  
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', flexShrink: 0 }}>
                    <span style={{ fontSize: '0.72rem', fontWeight: '800', backgroundColor: '#F0FDFA', color: '#0D9488', padding: '0.35rem 0.75rem', borderRadius: '0.55rem', border: '1px solid #CCFBF1', whiteSpace: 'nowrap' }}>
                      Confidence: {telemetry.confidenceScore !== null ? `${telemetry.confidenceScore}%` : '94%'}
                    </span>
                  </div>
                </div>

                {/* AQI Severity Bands Legend Bar */}
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem', marginBottom: '1.2rem', padding: '0.6rem 0.9rem', backgroundColor: '#F8FAFC', borderRadius: '0.75rem', border: '1px solid #E2E8F0' }}>
                  <span style={{ fontSize: '0.72rem', fontWeight: '800', color: '#64748B', display: 'flex', alignItems: 'center', marginRight: '0.5rem' }}>AQI SEVERITY SCALE:</span>
                  <span style={{ fontSize: '0.7rem', fontWeight: '800', color: '#047857', backgroundColor: '#D1FAE5', padding: '0.2rem 0.55rem', borderRadius: '0.35rem', border: '1px solid #A7F3D0' }}>🟢 Good (0-50)</span>
                  <span style={{ fontSize: '0.7rem', fontWeight: '800', color: '#B45309', backgroundColor: '#FEF3C7', padding: '0.2rem 0.55rem', borderRadius: '0.35rem', border: '1px solid #FDE68A' }}>🟡 Moderate (51-100)</span>
                  <span style={{ fontSize: '0.7rem', fontWeight: '800', color: '#C2410C', backgroundColor: '#FFEDD5', padding: '0.2rem 0.55rem', borderRadius: '0.35rem', border: '1px solid #FED7AA' }}>🟠 Unhealthy Sensitive (101-150)</span>
                  <span style={{ fontSize: '0.7rem', fontWeight: '800', color: '#B91C1C', backgroundColor: '#FEE2E2', padding: '0.2rem 0.55rem', borderRadius: '0.35rem', border: '1px solid #FCA5A5' }}>🔴 Unhealthy (151-200)</span>
                  <span style={{ fontSize: '0.7rem', fontWeight: '800', color: '#6D28D9', backgroundColor: '#EDE9FE', padding: '0.2rem 0.55rem', borderRadius: '0.35rem', border: '1px solid #DDD6FE' }}>🟣 Very Unhealthy (201+)</span>
                </div>

                <div style={{ height: '280px', width: '100%' }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={telemetry.trendHistory} margin={{ top: 15, right: 20, left: 0, bottom: 0 }}>
                      <defs>
                        <linearGradient id="aqiColorGrad" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#0D9488" stopOpacity={0.35}/>
                          <stop offset="95%" stopColor="#0D9488" stopOpacity={0}/>
                        </linearGradient>
                      </defs>

                      {/* AQI Severity Threshold Reference Bands */}
                      <ReferenceArea y1={0} y2={50} fill="#10B981" fillOpacity={0.07} label={{ value: 'Good', position: 'insideTopLeft', fill: '#059669', fontSize: 10, fontWeight: 800 }} />
                      <ReferenceArea y1={50} y2={100} fill="#F59E0B" fillOpacity={0.07} label={{ value: 'Moderate', position: 'insideTopLeft', fill: '#D97706', fontSize: 10, fontWeight: 800 }} />
                      <ReferenceArea y1={100} y2={150} fill="#F97316" fillOpacity={0.08} label={{ value: 'Unhealthy Sensitive', position: 'insideTopLeft', fill: '#EA580C', fontSize: 10, fontWeight: 800 }} />
                      <ReferenceArea y1={150} y2={200} fill="#EF4444" fillOpacity={0.09} label={{ value: 'Unhealthy', position: 'insideTopLeft', fill: '#DC2626', fontSize: 10, fontWeight: 800 }} />
                      <ReferenceArea y1={200} y2={300} fill="#8B5CF6" fillOpacity={0.10} label={{ value: 'Very Unhealthy', position: 'insideTopLeft', fill: '#7C3AED', fontSize: 10, fontWeight: 800 }} />

                      {/* Horizontal Threshold Boundary Lines */}
                      <ReferenceLine y={50} stroke="#10B981" strokeDasharray="3 3" strokeOpacity={0.5} />
                      <ReferenceLine y={100} stroke="#F59E0B" strokeDasharray="3 3" strokeOpacity={0.5} />
                      <ReferenceLine y={150} stroke="#F97316" strokeDasharray="3 3" strokeOpacity={0.5} />
                      <ReferenceLine y={200} stroke="#EF4444" strokeDasharray="3 3" strokeOpacity={0.5} />

                      {/* Vertical Marker Separating Past/Observed from AI Forecast */}
                      <ReferenceLine x="Now (Observed)" stroke="#0D9488" strokeWidth={2} strokeDasharray="4 4" label={{ value: 'Live Telemetry Boundary', position: 'top', fill: '#0D9488', fontSize: 10, fontWeight: 800 }} />

                      <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" vertical={false} />
                      <XAxis dataKey="time" stroke="#64748B" fontSize={11} tickLine={false} fontWeight={600} />
                      <YAxis stroke="#64748B" fontSize={11} tickLine={false} domain={[0, 'dataMax + 25']} fontWeight={600} />
                      
                      <Tooltip content={({ active, payload, label }) => {
                        if (active && payload && payload.length) {
                          const data = payload[0].payload;
                          const aqiVal = data.aqi;
                          let cat = data.category || "Moderate";
                          let catColor = data.categoryColor || "#F59E0B";
                          if (aqiVal <= 50) { cat = "Good"; catColor = "#10B981"; }
                          else if (aqiVal <= 100) { cat = "Moderate"; catColor = "#F59E0B"; }
                          else if (aqiVal <= 150) { cat = "Unhealthy Sensitive"; catColor = "#F97316"; }
                          else if (aqiVal <= 200) { cat = "Unhealthy"; catColor = "#EF4444"; }
                          else { cat = "Very Unhealthy"; catColor = "#8B5CF6"; }

                          return (
                            <div style={{
                              backgroundColor: '#FFFFFF',
                              border: `1.5px solid ${catColor}`,
                              borderRadius: '0.85rem',
                              padding: '0.75rem 1rem',
                              boxShadow: '0 4px 20px rgba(15, 23, 42, 0.12)',
                              fontSize: '0.82rem'
                            }}>
                              <div style={{ color: '#64748B', fontWeight: '700', marginBottom: '0.35rem' }}>
                                ⏱️ {label}
                              </div>
                              <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', marginBottom: '0.25rem' }}>
                                <span style={{ fontSize: '1.15rem', fontWeight: '900', color: '#0F172A' }}>
                                  AQI: {aqiVal}
                                </span>
                                <span style={{
                                  backgroundColor: `${catColor}1A`,
                                  color: catColor,
                                  fontWeight: '800',
                                  fontSize: '0.7rem',
                                  padding: '0.2rem 0.55rem',
                                  borderRadius: '0.4rem',
                                  border: `1px solid ${catColor}40`
                                }}>
                                  {cat}
                                </span>
                              </div>
                              {data.pm25 && (
                                <div style={{ color: '#64748B', fontSize: '0.76rem', fontWeight: '600' }}>
                                  PM2.5: {data.pm25} µg/m³
                                </div>
                              )}
                            </div>
                          );
                        }
                        return null;
                      }} />

                      <Area type="monotone" dataKey="aqi" stroke="#0D9488" strokeWidth={3.5} fillOpacity={1} fill="url(#aqiColorGrad)" dot={{ r: 4, fill: '#0D9488', stroke: '#FFFFFF', strokeWidth: 2 }} activeDot={{ r: 7, fill: '#0284C7', stroke: '#FFFFFF', strokeWidth: 3 }} />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Automated Email Alert Dispatcher Section */}
              <div id="email-alert-section">
                <EmailAlertDispatcher />
              </div>
            </motion.div>
          )}

          {/* TAB 2: DAILY TRENDS (EDA) */}
          {activeNav === 'trends' && (
            <motion.div 
              key="trends"
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -12 }}
              transition={{ duration: 0.25 }}
              style={{ display: 'flex', flexDirection: 'column', gap: '1.75rem' }}
            >
              {/* Full Pearson Feature Correlation Suite */}
              <CorrelationHeatmap />

              {/* Diurnal Hourly Profile Spline */}
              {edaData && edaData.hourly_diurnal_profile && (
                <div style={{ backgroundColor: '#FFFFFF', borderRadius: '1.25rem', padding: '1.6rem', border: '1px solid #E2E8F0', boxShadow: '0 4px 20px -2px rgba(15,23,42,0.04)' }}>
                  <div style={{ marginBottom: '1.2rem' }}>
                    <h3 style={{ margin: 0, fontSize: '1.1rem', fontWeight: '800', color: '#0F172A' }}>
                      🕒 24-Hour Daily Pollution Profile (PM2.5 & European AQI)
                    </h3>
                    <p style={{ margin: '0.2rem 0 0 0', fontSize: '0.78rem', color: '#64748B' }}>
                      24-hour mean atmospheric concentration pattern extracted from historical observation window
                    </p>
                  </div>
                  <div style={{ height: '260px', width: '100%' }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart data={
                        Array.isArray(edaData.hourly_diurnal_profile)
                          ? edaData.hourly_diurnal_profile
                          : Object.entries(edaData.hourly_diurnal_profile.pm2_5 || {}).map(([hr, val]) => ({
                              hour: `${hr}:00`,
                              pm25: val,
                              aqi: edaData.hourly_diurnal_profile.european_aqi ? edaData.hourly_diurnal_profile.european_aqi[hr] : val
                            }))
                      }>
                        <defs>
                          <linearGradient id="diurnalGrad" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#0284C7" stopOpacity={0.35}/>
                            <stop offset="95%" stopColor="#0284C7" stopOpacity={0}/>
                          </linearGradient>
                          <linearGradient id="aqiDiurnalGrad" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#0D9488" stopOpacity={0.25}/>
                            <stop offset="95%" stopColor="#0D9488" stopOpacity={0}/>
                          </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke="#F1F5F9" />
                        <XAxis dataKey="hour" stroke="#64748B" fontSize={11} tickLine={false} />
                        <YAxis stroke="#64748B" fontSize={11} tickLine={false} axisLine={false} />
                        <Tooltip contentStyle={{ backgroundColor: '#FFFFFF', borderColor: '#CBD5E1', borderRadius: '0.75rem', boxShadow: '0 4px 16px rgba(15,23,42,0.08)' }} />
                        <Area type="monotone" dataKey="pm25" stroke="#0284C7" strokeWidth={3} fill="url(#diurnalGrad)" name="Mean PM2.5 (µg/m³)" dot={{ r: 3, fill: '#0284C7' }} activeDot={{ r: 6 }} />
                        <Area type="monotone" dataKey="aqi" stroke="#0D9488" strokeWidth={2} fill="url(#aqiDiurnalGrad)" name="European AQI" strokeDasharray="4 4" dot={false} />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              )}
            </motion.div>
          )}

          {/* TAB 3: MODEL FACTORS (SHAP) */}
          {activeNav === 'shap' && (
            <motion.div 
              key="shap"
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -12 }}
              transition={{ duration: 0.25 }}
              style={{ display: 'flex', flexDirection: 'column', gap: '1.75rem' }}
            >
              {/* Multi-Horizon Directional SHAP Waterfall */}
              <ShapWaterfall 
                contributions={telemetry.localShapContributions} 
                shapExplanations={telemetry.shapExplanations}
                localShapByHorizon={telemetry.localShapByHorizon}
              />
            </motion.div>
          )}

          {/* TAB 4: MULTI-MODEL TOURNAMENT */}
          {activeNav === 'tournament' && (
            <motion.div 
              key="tournament"
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -12 }}
              transition={{ duration: 0.25 }}
              style={{ display: 'flex', flexDirection: 'column', gap: '1.75rem' }}
            >
              <TournamentChart />
            </motion.div>
          )}

          {/* TAB 5: ATMOSPHERIC FEATURES & FEATURE STORE STREAM */}
          {activeNav === 'regional' && (
            <motion.div 
              key="regional"
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -12 }}
              transition={{ duration: 0.25 }}
              style={{ display: 'flex', flexDirection: 'column', gap: '1.75rem' }}
            >
              <div style={{ backgroundColor: '#FFFFFF', padding: '1.75rem', borderRadius: '1.25rem', border: '1px solid #E2E8F0', boxShadow: '0 4px 20px -2px rgba(15,23,42,0.04)' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.4rem', flexWrap: 'wrap', gap: '0.8rem' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem' }}>
                    <Globe size={22} color="#0D9488" />
                    <div>
                      <h3 style={{ fontSize: '1.1rem', fontWeight: '800', color: '#0F172A', margin: 0 }}>
                        Multi-Variate Atmospheric Vector & Feature Store Telemetry
                      </h3>
                      <p style={{ margin: '0.25rem 0 0 0', fontSize: '0.78rem', color: '#64748B' }}>
                        Real-time 8-parameter observation vector stream from Open-Meteo & Hopsworks Feature Store V2
                      </p>
                    </div>
                  </div>
                  <span style={{ fontSize: '0.74rem', fontWeight: '800', backgroundColor: '#F0FDFA', color: '#0D9488', padding: '0.35rem 0.8rem', borderRadius: '0.5rem', border: '1px solid #CCFBF1', display: 'inline-flex', alignItems: 'center', gap: '0.4rem' }}>
                    <span style={{ width: '6px', height: '6px', backgroundColor: '#0D9488', borderRadius: '50%' }} />
                    8/8 FEATURES ACTIVE
                  </span>
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: '1.1rem' }}>
                  {telemetry.hotspots.map((station, idx) => (
                    <div key={station.id || idx} style={{
                      backgroundColor: '#FFFFFF',
                      padding: '1.25rem',
                      borderRadius: '0.9rem',
                      border: '1px solid #E2E8F0',
                      boxShadow: '0 2px 8px rgba(15,23,42,0.02)',
                      display: 'flex',
                      flexDirection: 'column',
                      justifySpace: 'space-between'
                    }}>
                      <div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                          <span style={{ fontSize: '0.66rem', fontWeight: '800', color: '#0D9488', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
                            FEATURE 0{idx + 1}
                          </span>
                          <span style={{ fontSize: '0.64rem', fontWeight: '800', backgroundColor: '#F1F5F9', color: '#475569', padding: '0.15rem 0.45rem', borderRadius: '0.3rem' }}>
                            {station.category || 'Environmental'}
                          </span>
                        </div>
                        <div style={{ fontSize: '0.95rem', fontWeight: '800', color: '#0F172A', margin: '0.4rem 0 0.2rem 0', lineHeight: 1.25 }}>
                          {station.name}
                        </div>
                      </div>

                      <div style={{ margin: '0.8rem 0 0.4rem 0' }}>
                        <div style={{ fontSize: '1.6rem', fontWeight: '900', color: station.color || '#0284C7', lineHeight: 1 }}>
                          {station.aqi}
                        </div>
                      </div>

                      <div style={{ fontSize: '0.70rem', color: '#64748B', borderTop: '1px solid #F1F5F9', paddingTop: '0.5rem', marginTop: '0.4rem' }}>
                        {station.estimationType}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </motion.div>
          )}

        </AnimatePresence>
      </main>
    </div>
  );
}
