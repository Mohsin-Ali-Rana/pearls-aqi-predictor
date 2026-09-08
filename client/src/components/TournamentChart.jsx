import React, { useState, useEffect } from 'react';
import { BarChart3, Trophy, Award } from 'lucide-react';

export default function TournamentChart() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  const getApiUrl = (path) => {
    if (typeof window !== 'undefined' && (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1')) {
      const envUrl = import.meta.env.VITE_API_URL || import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
      const cleanBase = envUrl.replace(/\/$/, '');
      return `${cleanBase}${path}`;
    }
    const envUrl = import.meta.env.VITE_API_URL || import.meta.env.VITE_API_BASE_URL || 'https://pearls-aqi-predictor-production.up.railway.app';
    const cleanBase = envUrl.replace(/\/$/, '');
    return `${cleanBase}${path}`;
  };

  useEffect(() => {
    fetch(getApiUrl('/api/tournament'))
      .then(res => res.json())
      .then(json => {
        setData(json);
        setLoading(false);
      })
      .catch(err => {
        console.error('Failed to load tournament data:', err);
        setLoading(false);
      });
  }, []);

  if (loading || !data || !data.horizons) {
    return (
      <div style={{ backgroundColor: '#FFFFFF', borderRadius: '1.25rem', padding: '1.6rem', border: '1px solid #E2E8F0', boxShadow: '0 4px 20px -2px rgba(15, 23, 42, 0.04)', color: '#64748B' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.2rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
            <BarChart3 size={20} color="#0D9488" />
            <div>
              <h3 style={{ margin: 0, fontSize: '1.1rem', fontWeight: '800', color: '#0F172A' }}>
                Multi-Model Tournament Matrix
              </h3>
              <p style={{ margin: '0.2rem 0 0 0', fontSize: '0.78rem', color: '#64748B' }}>
                Computing cross-validation RMSE scores for candidate estimators...
              </p>
            </div>
          </div>
          <span style={{ fontSize: '0.74rem', fontWeight: '800', backgroundColor: 'rgba(13,148,136,0.1)', color: '#0D9488', padding: '0.35rem 0.8rem', borderRadius: '0.5rem', border: '1px solid rgba(13,148,136,0.2)', display: 'inline-flex', alignItems: 'center', gap: '0.4rem' }}>
            <span className="pulse-dot" style={{ width: '6px', height: '6px', backgroundColor: '#0D9488', borderRadius: '50%' }} />
            COMPUTING MATRIX...
          </span>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.8rem' }}>
          {[1, 2, 3, 4].map(i => (
            <div key={i} style={{ height: '36px', width: '100%' }} className="skeleton-shimmer" />
          ))}
        </div>
      </div>
    );
  }

  // Extract estimators dynamically from live API payload
  const horizons = ['24h', '48h', '72h'];
  const estimators = (data && data.horizons && data.horizons['24h']) ? Object.keys(data.horizons['24h']) : [];

  return (
    <div style={{
      backgroundColor: '#FFFFFF',
      borderRadius: '1.25rem',
      padding: '1.6rem',
      border: '1px solid #E2E8F0',
      boxShadow: '0 4px 20px -2px rgba(15, 23, 42, 0.04)',
      color: '#0F172A'
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.4rem', flexWrap: 'wrap', gap: '0.8rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem' }}>
          <Award size={22} color="#0284C7" />
          <div>
            <h3 style={{ margin: 0, fontSize: '1.1rem', fontWeight: '800', color: '#0F172A', letterSpacing: '-0.01em' }}>
              Multi-Model Tournament Matrix (Per-Horizon RMSE & Winners)
            </h3>
            <p style={{ margin: '0.25rem 0 0 0', fontSize: '0.78rem', color: '#64748B' }}>
              Dynamic estimator competition over direct multi-horizons (+24h, +48h, +72h) evaluated by RMSE
            </p>
          </div>
        </div>
        {data.lifts && (
          <span style={{
            fontSize: '0.74rem',
            fontWeight: '800',
            color: '#059669',
            backgroundColor: '#ECFDF5',
            padding: '0.35rem 0.8rem',
            borderRadius: '0.5rem',
            border: '1px solid #A7F3D0'
          }}>
            Overall Lift: +{data.lifts.overall_lift_rmse_pct}%
          </span>
        )}
      </div>

      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.86rem', textAlign: 'left' }}>
          <thead>
            <tr style={{ borderBottom: '2px solid #F1F5F9', color: '#64748B', fontSize: '0.75rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              <th style={{ padding: '0.75rem 1rem' }}>Estimator</th>
              <th style={{ padding: '0.75rem 1rem' }}>Day 1 (24H RMSE)</th>
              <th style={{ padding: '0.75rem 1rem' }}>Day 2 (48H RMSE)</th>
              <th style={{ padding: '0.75rem 1rem' }}>Day 3 (72H RMSE)</th>
            </tr>
          </thead>
          <tbody>
            {estimators.map((est, idx) => {
              return (
                <tr key={est} style={{ borderBottom: '1px solid #F1F5F9', backgroundColor: idx % 2 === 0 ? '#FFFFFF' : '#FAFAFA' }}>
                  <td style={{ padding: '0.85rem 1rem', fontWeight: '800', color: '#0F172A' }}>
                    {est}
                  </td>
                  {horizons.map(h => {
                    const info = data.horizons[h] ? data.horizons[h][est] : null;
                    if (!info) return <td key={h} style={{ padding: '0.85rem 1rem', color: '#94A3B8' }}>-</td>;

                    const isWinner = info.winner;
                    return (
                      <td key={h} style={{ padding: '0.85rem 1rem' }}>
                        <span style={{
                          fontWeight: isWinner ? '900' : '600',
                          color: isWinner ? '#0284C7' : '#475569'
                        }}>
                          {info.rmse !== undefined ? info.rmse.toFixed(2) : '-'}
                        </span>
                        {isWinner && (
                          <span style={{ marginLeft: '0.5rem', fontSize: '0.72rem', fontWeight: '800', color: '#0284C7', backgroundColor: 'rgba(2,132,199,0.12)', padding: '0.2rem 0.55rem', borderRadius: '0.35rem', border: '1px solid rgba(2,132,199,0.25)', display: 'inline-flex', alignItems: 'center', gap: '0.3rem' }}>
                            <Trophy size={13} color="#0284C7" /> WINNER
                          </span>
                        )}
                      </td>
                    );
                  })}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
