import React, { useState } from 'react';
import { Sparkles } from 'lucide-react';

const FEATURE_META = {
  'pm2_5_lag_24h': { name: 'PM2.5 (24h Lag)', unit: 'µg/m³', desc: 'Prior Day Baseline' },
  'pm2_5_lag_1h': { name: 'PM2.5 (1h Lag)', unit: 'µg/m³', desc: 'Recent Persistence' },
  'pm2_5_lag_3h': { name: 'PM2.5 (3h Lag)', unit: 'µg/m³', desc: 'Short-term Trajectory' },
  'pm2_5_lag_12h': { name: 'PM2.5 (12h Lag)', unit: 'µg/m³', desc: 'Half-day Memory' },
  'pm10': { name: 'PM10 Coarse Particulates', unit: 'µg/m³', desc: 'Coarse Dust Stream' },
  'wind_speed_10m': { name: 'Wind Speed (10m)', unit: 'km/h', desc: 'Boundary Dispersion' },
  'relative_humidity_2m': { name: 'Relative Humidity', unit: '%', desc: 'Moisture Trap' },
  'temperature_2m': { name: 'Surface Temperature', unit: '°C', desc: 'Thermal Inversion' },
  'surface_pressure': { name: 'Surface Pressure', unit: 'hPa', desc: 'Barometric Trap' },
  'nitrogen_dioxide': { name: 'Nitrogen Dioxide (NO₂)', unit: 'µg/m³', desc: 'Traffic & Combustion' },
  'ozone': { name: 'Ground Ozone (O₃)', unit: 'µg/m³', desc: 'Photochemical Smog' },
  'european_aqi': { name: 'European AQI (EAQI)', unit: 'Index', desc: 'Regional Composite' },
  'sin_hour': { name: 'Diurnal Peak (Sine)', unit: '', desc: 'Traffic Rhythm' },
  'cos_hour': { name: 'Diurnal Cycle (Cosine)', unit: '', desc: 'Night Thermal Cycle' },
  'sin_day_of_week': { name: 'Weekly Rhythm (Sine)', unit: '', desc: 'Workday Activity' },
  'cos_day_of_week': { name: 'Weekly Cycle (Cosine)', unit: '', desc: 'Weekend Shift' },
  'pm2_5_rolling_24h_mean': { name: 'PM2.5 24h Mean', unit: 'µg/m³', desc: '24h Rolling Average' },
  'pm10_rolling_24h_mean': { name: 'PM10 24h Mean', unit: 'µg/m³', desc: 'Coarse Rolling Avg' },
  'aqi_change_rate': { name: 'AQI Acceleration', unit: 'Δ/h', desc: 'Rate of Change' }
};

export default function ShapWaterfall({ contributions, shapExplanations, localShapByHorizon }) {
  const [selectedHorizon, setSelectedHorizon] = useState('24h');

  // Retrieve SHAP vector for the active selected horizon
  let items = [];
  if (localShapByHorizon && typeof localShapByHorizon === 'object' && Array.isArray(localShapByHorizon[selectedHorizon]) && localShapByHorizon[selectedHorizon].length > 0) {
    items = localShapByHorizon[selectedHorizon];
  } else if (Array.isArray(contributions) && contributions.length > 0) {
    items = contributions;
  } else if (Array.isArray(shapExplanations) && shapExplanations.length > 0) {
    items = shapExplanations.map(s => ({
      feature: s?.feature || 'Feature',
      contribution: Number(s?.importance || 0),
      featureValue: null
    }));
  }

  if (!items || items.length === 0) {
    return (
      <div style={{ backgroundColor: '#FFFFFF', borderRadius: '1.25rem', padding: '1.6rem', border: '1px solid #E2E8F0', boxShadow: '0 4px 20px -2px rgba(15, 23, 42, 0.04)', color: '#64748B' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.2rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
            <Sparkles size={20} color="#0D9488" />
            <div>
              <h3 style={{ margin: 0, fontSize: '1.1rem', fontWeight: '800', color: '#0F172A' }}>
                Bi-Directional SHAP Feature Attribution
              </h3>
              <p style={{ margin: '0.2rem 0 0 0', fontSize: '0.78rem', color: '#64748B' }}>
                Evaluating TreeExplainer Shapley vectors for live observation X(t₀)...
              </p>
            </div>
          </div>
          <span style={{ fontSize: '0.74rem', fontWeight: '800', backgroundColor: 'rgba(13,148,136,0.1)', color: '#0D9488', padding: '0.35rem 0.8rem', borderRadius: '0.5rem', border: '1px solid rgba(13,148,136,0.2)', display: 'inline-flex', alignItems: 'center', gap: '0.4rem' }}>
            <span className="pulse-dot" style={{ width: '6px', height: '6px', backgroundColor: '#0D9488', borderRadius: '50%' }} />
            CALCULATING SHAP VECTORS...
          </span>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.8rem' }}>
          {[1, 2, 3, 4, 5, 6].map(i => (
            <div key={i} style={{ height: '36px', width: '100%' }} className="skeleton-shimmer" />
          ))}
        </div>
      </div>
    );
  }

  const sliceItems = items.slice(0, 8);
  const maxAbs = Math.max(...sliceItems.map(i => Math.abs(i.contribution || 0)), 0.001);

  const horizonLabels = {
    '24h': 'Day 1 (+24h Direct Prediction)',
    '48h': 'Day 2 (+48h Direct Prediction)',
    '72h': 'Day 3 (+72h Direct Prediction)'
  };

  return (
    <div style={{
      backgroundColor: '#FFFFFF',
      borderRadius: '1.25rem',
      padding: '1.6rem 1.8rem',
      border: '1px solid #E2E8F0',
      boxShadow: '0 4px 20px -2px rgba(15, 23, 42, 0.04)',
      color: '#0F172A'
    }}>
      {/* Header & Horizon Selector */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '1.2rem', gap: '1rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem' }}>
          <Sparkles size={22} color="#0D9488" />
          <div>
            <h3 style={{ margin: 0, fontSize: '1.15rem', fontWeight: '900', color: '#0F172A', letterSpacing: '-0.02em' }}>
              Bi-Directional SHAP Feature Attribution Waterfall
            </h3>
            <p style={{ margin: '0.25rem 0 0 0', fontSize: '0.78rem', color: '#64748B' }}>
              Signed impact (±Δ AQI) on <strong style={{ color: '#0F172A' }}>{horizonLabels[selectedHorizon]}</strong> state vector X(t₀)
            </p>
          </div>
        </div>

        {/* Horizon Selector Pill Bar */}
        <div style={{ display: 'flex', backgroundColor: '#F1F5F9', padding: '0.25rem', borderRadius: '0.75rem', border: '1px solid #E2E8F0', flexShrink: 0 }}>
          {[
            { id: '24h', label: 'Day 1 (+24h)' },
            { id: '48h', label: 'Day 2 (+48h)' },
            { id: '72h', label: 'Day 3 (+72h)' }
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setSelectedHorizon(tab.id)}
              style={{
                border: selectedHorizon === tab.id ? '1px solid #CBD5E1' : '1px solid transparent',
                backgroundColor: selectedHorizon === tab.id ? '#FFFFFF' : 'transparent',
                color: selectedHorizon === tab.id ? '#0F172A' : '#64748B',
                fontWeight: '700',
                fontSize: '0.76rem',
                padding: '0.4rem 0.85rem',
                borderRadius: '0.50rem',
                cursor: 'pointer',
                boxShadow: selectedHorizon === tab.id ? '0 2px 5px rgba(15,23,42,0.08)' : 'none',
                transition: 'all 0.15s ease-in-out',
                outline: 'none',
                whiteSpace: 'nowrap'
              }}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      {/* Axis Direction Legend Banner */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: '1fr auto 1fr',
        alignItems: 'center',
        backgroundColor: '#F8FAFC',
        padding: '0.65rem 1.25rem',
        borderRadius: '0.75rem',
        border: '1px solid #E2E8F0',
        marginBottom: '1.5rem',
        fontSize: '0.76rem',
        fontWeight: '800'
      }}>
        <div style={{ color: '#0D9488', display: 'flex', alignItems: 'center', gap: '0.4rem', justifyContent: 'flex-start' }}>
          <span>◄ Air Cleaning Effect</span>
          <span style={{ fontSize: '0.70rem', fontWeight: '600', color: '#64748B' }}>(-Δ AQI)</span>
        </div>

        <div style={{ color: '#475569', display: 'flex', alignItems: 'center', gap: '0.45rem', justifyContent: 'center' }}>
          <span style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: '#64748B' }} />
          <span>0.0 Neutral Baseline</span>
        </div>

        <div style={{ color: '#DC2626', display: 'flex', alignItems: 'center', gap: '0.4rem', justifyContent: 'flex-end' }}>
          <span style={{ fontSize: '0.70rem', fontWeight: '600', color: '#64748B' }}>(+Δ AQI)</span>
          <span>Pollution Inflation ►</span>
        </div>
      </div>

      {/* Diverging Bar Chart Container */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '1.15rem' }}>
        {sliceItems.map((item, idx) => {
          const cVal = item.contribution || 0;
          const isPositive = cVal >= 0;
          // Calculate bar length percentage relative to half width (max 34% to leave safe margin for value label)
          const barPct = Math.min((Math.abs(cVal) / maxAbs) * 34, 34);

          const meta = FEATURE_META[item.feature] || {
            name: item.feature.replace(/_/g, ' '),
            unit: '',
            desc: 'Feature Vector'
          };

          const displayObs = item.featureValue !== null && item.featureValue !== undefined
            ? `(${item.featureValue} ${meta.unit})`.trim()
            : '';

          return (
            <div key={idx} style={{ display: 'flex', alignItems: 'center', gap: '1.25rem' }}>
              {/* Feature Information Column */}
              <div style={{ width: '220px', flexShrink: 0, textAlign: 'right' }}>
                <div style={{ fontSize: '0.84rem', fontWeight: '800', color: '#0F172A', lineHeight: 1.25 }}>
                  {meta.name}
                </div>
                <div style={{ fontSize: '0.72rem', color: '#64748B', marginTop: '0.15rem' }}>
                  {displayObs && <span style={{ fontWeight: '700', color: '#334155', marginRight: '0.35rem' }}>{displayObs}</span>}
                  <span style={{ color: '#94A3B8' }}>{meta.desc}</span>
                </div>
              </div>

              {/* Bi-Directional Diverging Canvas (Central Axis at 50%) */}
              <div style={{
                flex: 1,
                height: '38px',
                backgroundColor: '#F8FAFC',
                borderRadius: '0.65rem',
                border: '1px solid #E2E8F0',
                position: 'relative',
                display: 'flex',
                alignItems: 'center',
                overflow: 'hidden'
              }}>
                {/* Dashed Central Axis (0.0 Line) */}
                <div style={{
                  position: 'absolute',
                  left: '50%',
                  top: 0,
                  bottom: 0,
                  width: '2px',
                  backgroundColor: '#94A3B8',
                  zIndex: 2,
                  boxShadow: '0 0 4px rgba(0,0,0,0.1)'
                }} />

                {/* NEGATIVE SHAP BAR (Extends LEFT from 50%) */}
                {!isPositive && (
                  <>
                    <div style={{
                      position: 'absolute',
                      right: '50%',
                      height: '22px',
                      width: `${barPct}%`,
                      background: 'linear-gradient(270deg, #0D9488 0%, #14B8A6 100%)',
                      borderRadius: '0.4rem 0 0 0.4rem',
                      transition: 'width 0.4s ease-in-out',
                      zIndex: 3,
                      boxShadow: '0 2px 6px rgba(13, 148, 136, 0.25)'
                    }} />
                    <span style={{
                      position: 'absolute',
                      right: `calc(50% + ${barPct}% + 8px)`,
                      fontSize: '0.78rem',
                      fontWeight: '900',
                      color: '#0D9488',
                      whiteSpace: 'nowrap',
                      zIndex: 4
                    }}>
                      {cVal.toFixed(1)} AQI
                    </span>
                  </>
                )}

                {/* POSITIVE SHAP BAR (Extends RIGHT from 50%) */}
                {isPositive && (
                  <>
                    <div style={{
                      position: 'absolute',
                      left: '50%',
                      height: '22px',
                      width: `${barPct}%`,
                      background: 'linear-gradient(90deg, #DC2626 0%, #EF4444 100%)',
                      borderRadius: '0 0.4rem 0.4rem 0',
                      transition: 'width 0.4s ease-in-out',
                      zIndex: 3,
                      boxShadow: '0 2px 6px rgba(220, 38, 38, 0.25)'
                    }} />
                    <span style={{
                      position: 'absolute',
                      left: `calc(50% + ${barPct}% + 8px)`,
                      fontSize: '0.78rem',
                      fontWeight: '900',
                      color: '#DC2626',
                      whiteSpace: 'nowrap',
                      zIndex: 4
                    }}>
                      +{cVal.toFixed(1)} AQI
                    </span>
                  </>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
