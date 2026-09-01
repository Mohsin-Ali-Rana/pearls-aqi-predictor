import React, { useState, useEffect } from 'react';
import { TrendingUp } from 'lucide-react';

export default function CorrelationHeatmap() {
  const [eda, setEda] = useState(null);
  const [loading, setLoading] = useState(true);
  const [activeView, setActiveView] = useState('rankings'); // 'rankings' or 'grid'

  useEffect(() => {
    fetch('/api/eda')
      .then(res => res.json())
      .then(json => {
        setEda(json);
        setLoading(false);
      })
      .catch(err => {
        console.error('Failed to load EDA summary:', err);
        setLoading(false);
      });
  }, []);

  if (loading || !eda) {
    return (
      <div style={{ backgroundColor: '#FFFFFF', borderRadius: '1.25rem', padding: '1.6rem', border: '1px solid #E2E8F0', boxShadow: '0 4px 20px -2px rgba(15, 23, 42, 0.04)', color: '#64748B' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.2rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
            <TrendingUp size={20} color="#0D9488" />
            <div>
              <h3 style={{ margin: 0, fontSize: '1.1rem', fontWeight: '800', color: '#0F172A' }}>
                Pearson Feature Correlation Suite
              </h3>
              <p style={{ margin: '0.2rem 0 0 0', fontSize: '0.78rem', color: '#64748B' }}>
                Querying feature correlation matrices from Hopsworks Feature Store...
              </p>
            </div>
          </div>
          <span style={{ fontSize: '0.74rem', fontWeight: '800', backgroundColor: 'rgba(13,148,136,0.1)', color: '#0D9488', padding: '0.35rem 0.8rem', borderRadius: '0.5rem', border: '1px solid rgba(13,148,136,0.2)', display: 'inline-flex', alignItems: 'center', gap: '0.4rem' }}>
            <span className="pulse-dot" style={{ width: '6px', height: '6px', backgroundColor: '#0D9488', borderRadius: '50%' }} />
            COMPUTING CORRELATIONS...
          </span>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.8rem' }}>
          {[1, 2, 3, 4, 5].map(i => (
            <div key={i} style={{ height: '32px', width: '100%' }} className="skeleton-shimmer" />
          ))}
        </div>
      </div>
    );
  }

  const targetCorrs = eda.target_correlations || {};
  const fullMatrix = eda.full_correlation_matrix || {};

  const sortedCorrs = Object.entries(targetCorrs)
    .filter(([k]) => k !== 'pm2_5')
    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));

  // Features list for 2D matrix
  const featureList = Object.keys(targetCorrs);

  const getCorrCategory = (val) => {
    const abs = Math.abs(val);
    const sign = val >= 0 ? 'Positive' : 'Negative';
    if (abs >= 0.7) return `Strong ${sign}`;
    if (abs >= 0.4) return `Moderate ${sign}`;
    if (abs >= 0.15) return `Weak ${sign}`;
    return 'Negligible';
  };

  const getCorrColor = (val) => {
    if (val >= 0.5) return '#0D9488'; // Deep teal
    if (val >= 0.2) return '#0284C7'; // Light teal/blue
    if (val > -0.2) return '#64748B'; // Gray neutral
    if (val > -0.5) return '#EA580C'; // Orange
    return '#DC2626'; // Red
  };

  return (
    <div style={{
      backgroundColor: '#FFFFFF',
      borderRadius: '1.25rem',
      padding: '1.6rem',
      border: '1px solid #E2E8F0',
      boxShadow: '0 4px 20px -2px rgba(15, 23, 42, 0.04)',
      color: '#0F172A'
    }}>
      {/* Header Bar */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.4rem', flexWrap: 'wrap', gap: '0.8rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem' }}>
          <TrendingUp size={22} color="#0D9488" />
          <div>
            <h3 style={{ margin: 0, fontSize: '1.15rem', fontWeight: '800', color: '#0F172A', letterSpacing: '-0.01em' }}>
              PM2.5 Feature Pearson Correlation Matrix & Dependencies
            </h3>
            <p style={{ margin: '0.25rem 0 0 0', fontSize: '0.78rem', color: '#64748B' }}>
              Derived from {eda.total_observations ? eda.total_observations.toLocaleString() : '1,464'} hourly observations in Hopsworks Feature Store V2
            </p>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '0.8rem' }}>
          {/* View Toggle */}
          <div style={{ display: 'flex', backgroundColor: '#F1F5F9', padding: '0.25rem', borderRadius: '0.75rem', border: '1px solid #E2E8F0' }}>
            <button
              onClick={() => setActiveView('rankings')}
              style={{
                border: 'none',
                backgroundColor: activeView === 'rankings' ? '#FFFFFF' : 'transparent',
                color: activeView === 'rankings' ? '#0F172A' : '#64748B',
                fontWeight: activeView === 'rankings' ? '800' : '600',
                fontSize: '0.76rem',
                padding: '0.4rem 0.85rem',
                borderRadius: '0.50rem',
                cursor: 'pointer',
                boxShadow: activeView === 'rankings' ? '0 2px 6px rgba(0,0,0,0.06)' : 'none',
                transition: 'all 0.2s ease'
              }}
            >
              Target Breakdown
            </button>
            <button
              onClick={() => setActiveView('grid')}
              style={{
                border: 'none',
                backgroundColor: activeView === 'grid' ? '#FFFFFF' : 'transparent',
                color: activeView === 'grid' ? '#0F172A' : '#64748B',
                fontWeight: activeView === 'grid' ? '800' : '600',
                fontSize: '0.76rem',
                padding: '0.4rem 0.85rem',
                borderRadius: '0.50rem',
                cursor: 'pointer',
                boxShadow: activeView === 'grid' ? '0 2px 6px rgba(0,0,0,0.06)' : 'none',
                transition: 'all 0.2s ease'
              }}
            >
              2D Heatmap Grid
            </button>
          </div>
        </div>
      </div>

      {/* VIEW 1: TARGET CORRELATION BREAKDOWN TABLE */}
      {activeView === 'rankings' && (
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem', textAlign: 'left' }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #F1F5F9', color: '#64748B', fontSize: '0.75rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                <th style={{ padding: '0.75rem 1rem' }}>Atmospheric Feature</th>
                <th style={{ padding: '0.75rem 1rem' }}>Pearson (r)</th>
                <th style={{ padding: '0.75rem 1rem' }}>Correlation Spectrum</th>
                <th style={{ padding: '0.75rem 1rem' }}>Relationship Strength</th>
              </tr>
            </thead>
            <tbody>
              {sortedCorrs.map(([feat, val], idx) => {
                const isPos = val >= 0;
                const barWidth = Math.min(Math.abs(val) * 100, 100);
                const color = getCorrColor(val);
                const category = getCorrCategory(val);

                return (
                  <tr key={feat} style={{ borderBottom: '1px solid #F1F5F9', backgroundColor: idx % 2 === 0 ? '#FFFFFF' : '#FAFAFA' }}>
                    <td style={{ padding: '0.85rem 1rem', fontWeight: '700', color: '#0F172A', fontFamily: 'monospace' }}>
                      {feat}
                    </td>
                    <td style={{ padding: '0.85rem 1rem', fontWeight: '900', color: color, fontSize: '0.92rem' }}>
                      {isPos ? `+${val.toFixed(4)}` : val.toFixed(4)}
                    </td>
                    <td style={{ padding: '0.85rem 1rem', width: '35%' }}>
                      <div style={{ height: '0.6rem', backgroundColor: '#E2E8F0', borderRadius: '0.3rem', overflow: 'hidden', position: 'relative' }}>
                        <div style={{
                          width: `${barWidth}%`,
                          height: '100%',
                          backgroundColor: color,
                          borderRadius: '0.3rem',
                          transition: 'width 0.4s ease'
                        }} />
                      </div>
                    </td>
                    <td style={{ padding: '0.85rem 1rem' }}>
                      <span style={{
                        fontSize: '0.72rem',
                        fontWeight: '800',
                        color: color,
                        backgroundColor: `${color}15`,
                        padding: '0.25rem 0.6rem',
                        borderRadius: '0.4rem',
                        border: `1px solid ${color}30`
                      }}>
                        {category}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {/* VIEW 2: 2D PAIRWISE HEATMAP GRID */}
      {activeView === 'grid' && (
        <div style={{ overflowX: 'auto', paddingTop: '0.5rem' }}>
          <table style={{ borderCollapse: 'collapse', fontSize: '0.78rem', margin: '0 auto' }}>
            <thead>
              <tr>
                <th style={{ padding: '0.5rem 0.8rem', color: '#94A3B8', fontSize: '0.7rem', verticalAlign: 'bottom', textAlign: 'right' }}>Feature</th>
                {featureList.map(f => (
                  <th key={f} style={{
                    padding: '0.5rem 0.2rem 0.8rem 0.2rem',
                    color: '#334155',
                    fontWeight: '800',
                    fontSize: '0.72rem',
                    textTransform: 'uppercase',
                    writingMode: 'vertical-rl',
                    transform: 'rotate(180deg)',
                    height: '120px',
                    verticalAlign: 'bottom',
                    textAlign: 'left'
                  }}>
                    {f}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {featureList.map(fRow => (
                <tr key={fRow}>
                  <td style={{ padding: '0.5rem 0.8rem', fontWeight: '800', color: '#334155', fontSize: '0.75rem', whiteSpace: 'nowrap', textAlign: 'right' }}>
                    {fRow}
                  </td>
                  {featureList.map(fCol => {
                    const val = fullMatrix[fRow] ? fullMatrix[fRow][fCol] : (fRow === fCol ? 1.0 : (targetCorrs[fCol] || 0));
                    const numVal = typeof val === 'number' ? val : 0;
                    const isSelf = fRow === fCol;
                    const bgAlpha = Math.abs(numVal);
                    
                    let bgColor = '#F1F5F9';
                    let textColor = '#0F172A';
                    if (isSelf) {
                      bgColor = '#CBD5E1';
                      textColor = '#0F172A';
                    } else if (numVal > 0) {
                      bgColor = `rgba(13, 148, 136, ${Math.max(0.12, bgAlpha * 0.75)})`;
                      textColor = bgAlpha > 0.4 ? '#042F2C' : '#0F172A';
                    } else if (numVal < 0) {
                      bgColor = `rgba(220, 38, 38, ${Math.max(0.12, bgAlpha * 0.75)})`;
                      textColor = bgAlpha > 0.4 ? '#450A0A' : '#0F172A';
                    }

                    return (
                      <td key={fCol} style={{
                        padding: '0.5rem 0.6rem',
                        textAlign: 'center',
                        backgroundColor: bgColor,
                        color: textColor,
                        fontWeight: '800',
                        border: '1px solid #FFFFFF',
                        borderRadius: '0.25rem',
                        fontSize: '0.74rem'
                      }}>
                        {numVal.toFixed(2)}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
