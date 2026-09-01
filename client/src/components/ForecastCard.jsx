import React from 'react';
import { Thermometer, Droplets, Wind, Activity, Clock } from 'lucide-react';

export default function ForecastCard({ forecast, isLoading }) {
  if (!forecast && !isLoading) return null;

  const fc = forecast || {};
  const isCardLoading = isLoading || fc.aqi === null || fc.aqi === undefined;

  const safeNumber = (val) => {
    if (val === null || val === undefined || val === '') return null;
    const n = Number(val);
    return isNaN(n) ? null : n;
  };

  const aqiNum = safeNumber(fc.aqi);
  const aqi = isCardLoading || aqiNum === null ? '--' : Math.round(aqiNum);
  const status = isCardLoading ? 'CALCULATING...' : (fc.status || 'Good AQI');
  const color = isCardLoading ? '#64748B' : (fc.color || '#10B981');
  const horizon = fc.horizon || '24H';

  // Clean Model Name (Requirement 2: No 'Winner' text, clearly visible)
  let rawModelName = isCardLoading ? 'Direct Engine' : (fc.modelName || 'LightGBM');
  const cleanModelName = String(rawModelName).replace(/Winner/gi, '').trim() || 'LightGBM';

  const rmseNum = safeNumber(fc.rmse);
  const rmse = !isCardLoading && rmseNum !== null ? `±${rmseNum.toFixed(2)} RMSE` : '-- RMSE';
  const rawTargetTs = isCardLoading ? 'Syncing stream...' : (fc.targetTimestamp || 'Target Horizon');
  // Format target timestamp cleanly showing full date and 24h time (e.g. "Sep 02, 23:00")
  const targetTs = String(rawTargetTs).replace(/\s*UTC/gi, '').trim();

  const tempNum = safeNumber(fc.temperature);
  const temp = !isCardLoading && tempNum !== null ? `${tempNum.toFixed(1)}°C` : '--°C';

  const humidityNum = safeNumber(fc.humidity);
  const humidity = !isCardLoading && humidityNum !== null ? `${humidityNum.toFixed(0)}%` : '--%';

  const windNum = safeNumber(fc.windSpeed);
  const wind = !isCardLoading && windNum !== null ? `${windNum.toFixed(1)} km/h` : '-- km/h';

  const pm25Num = safeNumber(fc.pm25) ?? safeNumber(fc.predicted_pm2_5);
  const pm25Val = !isCardLoading && pm25Num !== null ? `${pm25Num.toFixed(1)} µg/m³` : '-- µg/m³';

  const dayLabel = horizon === '24H' ? 'Day 1 (+24h)' : horizon === '48H' ? 'Day 2 (+48h)' : 'Day 3 (+72h)';

  return (
    <div style={{
      backgroundColor: '#FFFFFF',
      borderRadius: '1.25rem',
      border: '1px solid #E2E8F0',
      boxShadow: '0 4px 20px -2px rgba(15, 23, 42, 0.05)',
      display: 'flex',
      flexDirection: 'column',
      width: '100%',
      boxSizing: 'border-box',
      overflow: 'hidden'
    }}>
      
      {/* 1. Top Front Header Banner Bar (Teal Brand Gradient) */}
      <div style={{
        background: 'linear-gradient(135deg, #0F766E 0%, #0D9488 50%, #14B8A6 100%)',
        padding: '0.65rem 0.9rem',
        display: 'flex',
        justify: 'space-between',
        alignItems: 'center',
        color: '#FFFFFF',
        position: 'relative',
        minHeight: '38px',
        width: '100%',
        boxSizing: 'border-box'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', fontSize: '0.74rem', fontWeight: '800', letterSpacing: '0.01em', whiteSpace: 'nowrap', flexShrink: 0 }}>
          <Clock size={14} color="#FFFFFF" style={{ flexShrink: 0 }} />
          <span>{dayLabel}</span>
        </div>
        
        {/* Floating Glassmorphism Pill Badge Pinned at Far Top Right Corner */}
        <div style={{
          marginLeft: 'auto',
          position: 'relative',
          backgroundColor: 'rgba(255, 255, 255, 0.22)',
          backdropFilter: 'blur(6px)',
          border: '1px solid rgba(255, 255, 255, 0.4)',
          borderRadius: '9999px',
          padding: '0.15rem 0.55rem',
          display: 'inline-flex',
          alignItems: 'center',
          gap: '0.3rem',
          fontSize: '0.64rem',
          fontWeight: '800',
          color: '#FFFFFF',
          whiteSpace: 'nowrap',
          flexShrink: 0
        }}>
          <span style={{ width: '4px', height: '4px', borderRadius: '50%', backgroundColor: '#34D399', flexShrink: 0 }} />
          <span>{cleanModelName}</span>
          <span style={{
            position: 'absolute',
            top: '-5px',
            right: '-3px',
            backgroundColor: '#F59E0B',
            color: '#0F172A',
            fontSize: '0.45rem',
            fontWeight: '900',
            padding: '0.03rem 0.25rem',
            borderRadius: '0.2rem',
            letterSpacing: '0.04em',
            boxShadow: '0 2px 4px rgba(0,0,0,0.18)',
            lineHeight: 1
          }}>
            BEST
          </span>
        </div>
      </div>

      {/* 2. Card Body Content */}
      <div style={{
        padding: '1rem',
        display: 'flex',
        flexDirection: 'column',
        gap: '0.85rem'
      }}>
        {/* Hero AQI Score & Dynamic PM2.5 Badge */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.2rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: '0.63rem', fontWeight: '800', color: '#64748B', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              Forecasted AQI
            </span>
            <span style={{ fontSize: '0.66rem', fontWeight: '800', color: '#1D4ED8', backgroundColor: '#DBEAFE', padding: '0.18rem 0.55rem', borderRadius: '9999px' }}>
              PM₂.₅: {pm25Val}
            </span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', flexDirection: 'column', gap: '0.45rem', margin: '0.2rem 0' }}>
            {isCardLoading ? (
              <div style={{ width: '80px', height: '38px' }} className="skeleton-shimmer" />
            ) : (
              <span style={{ fontSize: '2.8rem', fontWeight: '900', color: '#0F172A', lineHeight: 1, letterSpacing: '-0.04em' }}>
                {aqi}
              </span>
            )}
            <span style={{
              fontSize: '0.7rem',
              fontWeight: '800',
              color: '#B45309',
              backgroundColor: '#FEF3C7',
              padding: '0.25rem 0.7rem',
              borderRadius: '9999px',
              border: '1px solid #FDE68A',
              display: 'inline-flex',
              alignItems: 'center',
              gap: '0.35rem',
              lineHeight: 1.2
            }}>
              <span style={{ width: '5px', height: '5px', borderRadius: '50%', backgroundColor: '#D97706', flexShrink: 0 }} />
              {status}
            </span>
          </div>
        </div>

        {/* Dual Telemetry Sub-Cards (Confidence & Target) - Absolute Grid Centered Icons */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem' }}>
          
          {/* Confidence Sub-Card */}
          <div style={{
            backgroundColor: '#F8FAFC',
            borderRadius: '0.75rem',
            padding: '0.55rem 0.65rem',
            border: '1px solid #E2E8F0',
            display: 'flex',
            flexDirection: 'column',
            gap: '0.25rem',
            minWidth: 0
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
              <div style={{
                width: '22px',
                height: '22px',
                borderRadius: '50%',
                backgroundColor: '#ECFDF5',
                border: '1px solid #A7F3D0',
                display: 'grid',
                placeItems: 'center',
                flexShrink: 0
              }}>
                <Activity size={12} color="#059669" style={{ margin: 0, display: 'block' }} />
              </div>
              <span style={{ fontSize: '0.62rem', fontWeight: '800', color: '#475569', textTransform: 'uppercase', letterSpacing: '0.03em', lineHeight: 1 }}>
                Confidence
              </span>
            </div>
            <div style={{ fontSize: '0.78rem', fontWeight: '800', color: '#0F172A', whiteSpace: 'nowrap', paddingLeft: '0.1rem' }}>
              {rmse}
            </div>
          </div>

          {/* Target Sub-Card */}
          <div style={{
            backgroundColor: '#F8FAFC',
            borderRadius: '0.75rem',
            padding: '0.55rem 0.65rem',
            border: '1px solid #E2E8F0',
            display: 'flex',
            flexDirection: 'column',
            gap: '0.25rem',
            minWidth: 0
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
              <div style={{
                width: '22px',
                height: '22px',
                borderRadius: '50%',
                backgroundColor: '#EFF6FF',
                border: '1px solid #BFDBFE',
                display: 'grid',
                placeItems: 'center',
                flexShrink: 0
              }}>
                <Clock size={12} color="#2563EB" style={{ margin: 0, display: 'block' }} />
              </div>
              <span style={{ fontSize: '0.62rem', fontWeight: '800', color: '#475569', textTransform: 'uppercase', letterSpacing: '0.03em', lineHeight: 1 }}>
                Target
              </span>
            </div>
            <div style={{ fontSize: '0.76rem', fontWeight: '800', color: '#0F172A', whiteSpace: 'nowrap', paddingLeft: '0.1rem' }}>
              {targetTs}
            </div>
          </div>

        </div>

        {/* Forecasted Atmospheric Covariates Section with Vertical Dividers */}
        <div style={{
          backgroundColor: '#F8FAFC',
          borderRadius: '0.75rem',
          padding: '0.65rem 0.75rem',
          border: '1px solid #E2E8F0',
          display: 'flex',
          flexDirection: 'column',
          gap: '0.4rem'
        }}>
          <div style={{ fontSize: '0.62rem', fontWeight: '800', color: '#475569', textTransform: 'uppercase', letterSpacing: '0.05em', textAlign: 'center' }}>
            Forecasted Weather Covariates
          </div>
          {isCardLoading ? (
            <div style={{ width: '100%', height: '18px' }} className="skeleton-shimmer" />
          ) : (
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '0.76rem', fontWeight: '800', color: '#0F172A' }}>
              <span style={{ display: 'inline-flex', alignItems: 'center', gap: '0.3rem', flex: 1, justifyContent: 'center' }}>
                <Thermometer size={14} color="#EA580C" /> {temp}
              </span>
              <div style={{ width: '1px', height: '14px', backgroundColor: '#E2E8F0' }} />
              <span style={{ display: 'inline-flex', alignItems: 'center', gap: '0.3rem', flex: 1, justifyContent: 'center' }}>
                <Droplets size={14} color="#0284C7" /> {humidity}
              </span>
              <div style={{ width: '1px', height: '14px', backgroundColor: '#E2E8F0' }} />
              <span style={{ display: 'inline-flex', alignItems: 'center', gap: '0.3rem', flex: 1, justifyContent: 'center' }}>
                <Wind size={14} color="#0D9488" /> {wind}
              </span>
            </div>
          )}
        </div>
      </div>

    </div>
  );
}
