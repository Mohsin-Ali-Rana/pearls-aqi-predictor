import React from 'react';
import { Trophy, Thermometer, Droplets, Gauge, Wind, ShieldCheck, Activity, Clock, TrendingUp, TrendingDown, Minus } from 'lucide-react';

export default function HeroAQIGauge({ telemetry, isLoading }) {
  if (!telemetry) return null;

  const isDataLoading = isLoading || telemetry.currentAQI === null || telemetry.currentAQI === undefined;

  const currentAQI = isDataLoading ? null : telemetry.currentAQI;
  const pm25 = isDataLoading ? '--' : telemetry.pm25;
  const aqiColor = isDataLoading ? '#64748B' : (telemetry.aqiColor || '#0D9488');
  const aqiStatus = isDataLoading ? 'LOADING TELEMETRY...' : (telemetry.aqiStatus || 'Good');
  
  // Dynamic Real Timestamp & Delta Metrics from Live Backend Telemetry
  const formattedTime = (telemetry && (telemetry.last_updated || telemetry.lastUpdated))
    ? (telemetry.last_updated || telemetry.lastUpdated) 
    : new Date().toLocaleDateString('en-US', { month: 'short', day: '2-digit', year: 'numeric' }) + 
      ' at ' + 
      new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) + ' PKT';
  const deltaPct = telemetry.aqi_delta_pct !== undefined ? telemetry.aqi_delta_pct : 0.0;
  
  // Dynamic Weather Covariates from Live Backend Telemetry
  const cw = telemetry.currentWeather || {};
  const fc24 = (telemetry.forecasts && telemetry.forecasts[0]) || {};
  
  const tempVal = !isDataLoading && cw.temperature !== undefined ? cw.temperature : (!isDataLoading && fc24.temperature ? fc24.temperature : '--');
  const humidityVal = !isDataLoading && cw.humidity !== undefined ? cw.humidity : (!isDataLoading && fc24.humidity ? fc24.humidity : '--');
  const pressureVal = !isDataLoading && cw.pressure !== undefined ? cw.pressure : (!isDataLoading && fc24.pressure ? fc24.pressure : '--');
  const windVal = !isDataLoading && cw.wind_speed !== undefined ? cw.wind_speed : (!isDataLoading && fc24.windSpeed ? fc24.windSpeed : '--');

  const windDir = !isDataLoading && cw.wind_direction ? cw.wind_direction : 'Light Breeze';
  const boundaryCond = !isDataLoading && cw.boundary_condition ? cw.boundary_condition : 'Stable Boundary Layer';
  const aerosolRisk = !isDataLoading && cw.aerosol_risk ? cw.aerosol_risk : 'Low Aerosol Trap';
  const inversionRisk = !isDataLoading && cw.inversion_risk ? cw.inversion_risk : 'Standard Barometric';

  // Dynamic Persistence Lift Metrics
  const lift = telemetry.persistenceLift || {};
  const d1Lift = !isDataLoading && lift.day1_lift_pct !== undefined ? `+${lift.day1_lift_pct}%` : '--%';
  const d1Rmse = !isDataLoading && lift.day1_rmse !== undefined ? lift.day1_rmse : '--';
  const d2Lift = !isDataLoading && lift.day2_lift_pct !== undefined ? `+${lift.day2_lift_pct}%` : '--%';
  const d2Rmse = !isDataLoading && lift.day2_rmse !== undefined ? lift.day2_rmse : '--';
  const d3Lift = !isDataLoading && lift.day3_lift_pct !== undefined ? `+${lift.day3_lift_pct}%` : '--%';
  const d3Rmse = !isDataLoading && lift.day3_rmse !== undefined ? lift.day3_rmse : '--';

  // Calculate arc rotation percentage for gauge (max AQI 300)
  const gaugePct = isDataLoading ? 0 : Math.min((currentAQI || 0) / 300, 1.0);
  const rotationDeg = isDataLoading ? 0 : gaugePct * 180;

  // Dynamic status pill background and border colors matching standard AQI thresholds
  const getStatusBadgeStyle = (aqi) => {
    if (aqi === null || aqi === undefined) return { bg: '#F1F5F9', border: '#E2E8F0', text: '#64748B' };
    if (aqi <= 50) return { bg: '#F0FDFA', border: '#CCFBF1', text: '#0D9488' };
    if (aqi <= 100) return { bg: '#FEF3C7', border: '#FDE68A', text: '#D97706' };
    if (aqi <= 150) return { bg: '#FFF7ED', border: '#FFEDD5', text: '#EA580C' };
    if (aqi <= 200) return { bg: '#FEF2F2', border: '#FCA5A5', text: '#DC2626' };
    return { bg: '#F5F3FF', border: '#DDD6FE', text: '#7C3AED' };
  };
  const statusStyle = getStatusBadgeStyle(currentAQI);

  // SVG Arc Parameters for High-Precision Rendering
  const strokeWidth = 14;
  const radius = 80;
  const arcLength = Math.PI * radius; // ~251.327
  const validAQI = currentAQI || 0;
  const pct = isDataLoading ? 0 : Math.min(Math.max(validAQI / 300, 0), 1.0);
  const dashOffset = arcLength * (1 - pct);
  
  // Indicator dot position on the arc circumference
  const angleRad = Math.PI * (1 - pct);
  const dotX = 100 + radius * Math.cos(angleRad);
  const dotY = 95 - radius * Math.sin(angleRad);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
      
      {/* SECTION 1: Top Hero Grid (Gauge Left + 2x2 Weather Grid Right) */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(340px, 1fr))', gap: '1.25rem', alignItems: 'stretch' }}>
        
        {/* Left Card: Air Quality Index Arc Gauge */}
        <div style={{
          backgroundColor: '#FFFFFF',
          borderRadius: '1.25rem',
          padding: '1.5rem',
          border: '1px solid #E2E8F0',
          boxShadow: '0 4px 20px -2px rgba(15, 23, 42, 0.04)',
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'space-between',
          color: '#0F172A'
        }}>
          {/* Header */}
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: '0.8rem', fontWeight: '800', color: '#64748B', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              Current Air Quality Index
            </span>
            <span style={{
              fontSize: '0.7rem',
              fontWeight: '800',
              backgroundColor: isDataLoading ? '#F1F5F9' : 'rgba(13,148,136,0.1)',
              color: isDataLoading ? '#64748B' : '#0D9488',
              padding: '0.25rem 0.6rem',
              borderRadius: '0.4rem',
              border: isDataLoading ? '1px solid #CBD5E1' : '1px solid rgba(13,148,136,0.2)',
              display: 'inline-flex',
              alignItems: 'center',
              gap: '0.35rem'
            }}>
              <span className="pulse-dot" style={{ width: '6px', height: '6px', backgroundColor: isDataLoading ? '#64748B' : '#10B981', borderRadius: '50%', boxShadow: isDataLoading ? 'none' : '0 0 6px #10B981' }} />
              {isDataLoading ? 'FETCHING TELEMETRY...' : 'LIVE TELEMETRY'}
            </span>
          </div>

          {/* Semi-Circle Gauge Graphic & Live AQI Value */}
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', margin: '0.8rem 0 0.4rem 0' }}>
            
            {/* SVG Arc Gauge with Center AQI Number */}
            <div style={{ position: 'relative', width: '220px', height: '110px', display: 'flex', justifyContent: 'center' }}>
              <svg width="220" height="110" viewBox="0 0 200 100" style={{ overflow: 'visible' }}>
                {/* Background Track Arc */}
                <path
                  d="M 20 90 A 80 80 0 0 1 180 90"
                  fill="none"
                  stroke="#F1F5F9"
                  strokeWidth="14"
                  strokeLinecap="round"
                />
                
                {/* Active Dynamic AQI Arc */}
                <path
                  d="M 20 90 A 80 80 0 0 1 180 90"
                  fill="none"
                  stroke={isDataLoading ? '#CBD5E1' : aqiColor}
                  strokeWidth="14"
                  strokeLinecap="round"
                  strokeDasharray={arcLength}
                  strokeDashoffset={dashOffset}
                  style={{ transition: 'stroke-dashoffset 1s ease, stroke 0.5s ease' }}
                />

                {/* Glowing Tip Indicator Dot */}
                {!isDataLoading && validAQI > 0 && (
                  <circle
                    cx={dotX}
                    cy={dotY - 5}
                    r="6.5"
                    fill="#FFFFFF"
                    stroke={aqiColor}
                    strokeWidth="3"
                    style={{
                      transition: 'cx 1s ease, cy 1s ease',
                      filter: 'drop-shadow(0px 2px 4px rgba(0,0,0,0.12))'
                    }}
                  />
                )}

                {/* Scale Ticks at ends (0, 300+) */}
                <text x="12" y="102" fontSize="9.5" fontWeight="800" fill="#94A3B8" textAnchor="middle">0</text>
                <text x="188" y="102" fontSize="9.5" fontWeight="800" fill="#94A3B8" textAnchor="middle">300+</text>
              </svg>

              {/* Big Centered AQI Number Inside Arc */}
              <div style={{
                position: 'absolute',
                bottom: '10px',
                left: 0,
                right: 0,
                textAlign: 'center',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center'
              }}>
                {isDataLoading ? (
                  <div style={{ width: '80px', height: '38px' }} className="skeleton-shimmer" />
                ) : (
                  <span style={{ fontSize: '3.2rem', fontWeight: '900', color: '#0F172A', lineHeight: 1, letterSpacing: '-0.04em' }}>
                    {Math.round(currentAQI)}
                  </span>
                )}
              </div>
            </div>

            {/* Executive Status Pill Badge BELOW the Arc Gauge */}
            {!isDataLoading && (
              <div style={{
                fontSize: '0.76rem',
                fontWeight: '800',
                color: statusStyle.text,
                backgroundColor: statusStyle.bg,
                border: `1px solid ${statusStyle.border}`,
                padding: '0.35rem 0.9rem',
                borderRadius: '2rem',
                textTransform: 'uppercase',
                letterSpacing: '0.04em',
                marginTop: '0.9rem',
                boxShadow: '0 2px 6px -1px rgba(0,0,0,0.03)',
                textAlign: 'center',
                maxWidth: '90%'
              }}>
                {aqiStatus}
              </div>
            )}
          </div>

          {/* Executive Card Footer */}
          <div style={{
            display: 'flex',
            flexDirection: 'column',
            gap: '0.5rem',
            paddingTop: '0.75rem',
            borderTop: '1px solid #F1F5F9'
          }}>
            {/* Row 1: Baseline PM2.5 Sensor Reading */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.45rem' }}>
                <Activity size={15} color="#0284C7" />
                <span style={{ color: '#64748B', fontWeight: '600', fontSize: '0.8rem' }}>
                  Baseline Sensor: <strong style={{ color: '#0F172A', marginLeft: '0.2rem' }}>PM2.5</strong>
                </span>
              </div>
              {isDataLoading ? (
                <div style={{ width: '60px', height: '16px' }} className="skeleton-shimmer" />
              ) : (
                <div style={{ display: 'flex', alignItems: 'baseline', gap: '0.25rem' }}>
                  <span style={{ fontWeight: '900', color: '#0284C7', fontSize: '1rem' }}>{pm25}</span>
                  <span style={{ fontWeight: '700', color: '#64748B', fontSize: '0.75rem' }}>µg/m³</span>
                </div>
              )}
            </div>

            {/* Row 2: Exact Clock & Date Timestamp */}
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.45rem',
              color: '#64748B',
              fontSize: '0.78rem',
              fontWeight: '500',
              paddingTop: '0.45rem',
              borderTop: '1px solid #F8FAFC'
            }}>
              <Clock size={14} color="#0D9488" style={{ flexShrink: 0 }} />
              <span>
                Last updated: <strong style={{ color: '#0F172A', fontWeight: '700', marginLeft: '0.25rem' }}>{formattedTime}</strong>
              </span>
            </div>

            {/* Row 3: vs last observation delta % badge */}
            <div style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              paddingTop: '0.45rem',
              borderTop: '1px solid #F1F5F9',
              fontSize: '0.78rem',
              gap: '0.75rem'
            }}>
              <span style={{ color: '#64748B', fontWeight: '600', fontSize: '0.76rem', whiteSpace: 'nowrap' }}>
                vs last observation
              </span>
              {isDataLoading ? (
                <div style={{ width: '56px', height: '20px' }} className="skeleton-shimmer" />
              ) : (
                <span style={{
                  fontWeight: '800',
                  fontSize: '0.75rem',
                  color: deltaPct > 0 ? '#DC2626' : (deltaPct < 0 ? '#0D9488' : '#475569'),
                  backgroundColor: deltaPct > 0 ? '#FEE2E2' : (deltaPct < 0 ? '#F0FDFA' : '#F1F5F9'),
                  padding: '0.2rem 0.65rem',
                  borderRadius: '0.45rem',
                  border: `1px solid ${deltaPct > 0 ? '#FCA5A5' : (deltaPct < 0 ? '#CCFBF1' : '#E2E8F0')}`,
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: '0.3rem',
                  whiteSpace: 'nowrap'
                }}>
                  {deltaPct > 0 ? (
                    <>
                      <TrendingUp size={12} color="#DC2626" />
                      <span>↑ +{deltaPct}%</span>
                    </>
                  ) : deltaPct < 0 ? (
                    <>
                      <TrendingDown size={12} color="#0D9488" />
                      <span>↓ {deltaPct}%</span>
                    </>
                  ) : (
                    <span>0.0%</span>
                  )}
                </span>
              )}
            </div>
          </div>
        </div>

        {/* Right Card: 2x2 Meteorological Covariates Grid */}
        <div style={{
          backgroundColor: '#FFFFFF',
          borderRadius: '1.25rem',
          padding: '1.5rem',
          border: '1px solid #E2E8F0',
          boxShadow: '0 4px 20px -2px rgba(15, 23, 42, 0.04)',
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'space-between'
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
            <span style={{ fontSize: '0.8rem', fontWeight: '800', color: '#64748B', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              Atmospheric Covariates
            </span>
            <span style={{ fontSize: '0.7rem', fontWeight: '700', color: '#0284C7', backgroundColor: '#F0F9FF', padding: '0.2rem 0.5rem', borderRadius: '0.4rem', border: '1px solid #BAE6FD' }}>
              LIVE TELEMETRY STREAM
            </span>
          </div>

          {/* 2x2 Grid of Spacious Weather Cards */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '1rem', flex: 1 }}>
            
            {/* TEMPERATURE */}
            <div style={{ backgroundColor: '#F8FAFC', padding: '1rem', borderRadius: '0.85rem', border: '1px solid #F1F5F9', display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <div style={{ backgroundColor: '#FFEDD5', padding: '0.35rem', borderRadius: '0.4rem', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <Thermometer size={15} color="#EA580C" />
                </div>
                <span style={{ fontSize: '0.7rem', fontWeight: '800', color: '#64748B', textTransform: 'uppercase' }}>TEMPERATURE</span>
              </div>
              <div style={{ margin: '0.5rem 0' }}>
                <span style={{ fontSize: '1.5rem', fontWeight: '900', color: '#0F172A' }}>{tempVal}</span>
                <span style={{ fontSize: '0.85rem', fontWeight: '700', color: '#64748B', marginLeft: '0.15rem' }}>°C</span>
              </div>
              <span style={{ fontSize: '0.72rem', color: '#0D9488', fontWeight: '700' }}>{boundaryCond}</span>
            </div>

            {/* HUMIDITY */}
            <div style={{ backgroundColor: '#F8FAFC', padding: '1rem', borderRadius: '0.85rem', border: '1px solid #F1F5F9', display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <div style={{ backgroundColor: '#E0F2FE', padding: '0.35rem', borderRadius: '0.4rem', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <Droplets size={15} color="#0284C7" />
                </div>
                <span style={{ fontSize: '0.7rem', fontWeight: '800', color: '#64748B', textTransform: 'uppercase' }}>HUMIDITY</span>
              </div>
              <div style={{ margin: '0.5rem 0' }}>
                <span style={{ fontSize: '1.5rem', fontWeight: '900', color: '#0F172A' }}>{humidityVal}</span>
                <span style={{ fontSize: '0.85rem', fontWeight: '700', color: '#64748B', marginLeft: '0.15rem' }}>%</span>
              </div>
              <span style={{ fontSize: '0.72rem', color: '#D97706', fontWeight: '700' }}>{aerosolRisk}</span>
            </div>

            {/* PRESSURE */}
            <div style={{ backgroundColor: '#F8FAFC', padding: '1rem', borderRadius: '0.85rem', border: '1px solid #F1F5F9', display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <div style={{ backgroundColor: '#F1F5F9', padding: '0.35rem', borderRadius: '0.4rem', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <Gauge size={15} color="#475569" />
                </div>
                <span style={{ fontSize: '0.7rem', fontWeight: '800', color: '#64748B', textTransform: 'uppercase' }}>PRESSURE</span>
              </div>
              <div style={{ margin: '0.5rem 0' }}>
                <span style={{ fontSize: '1.5rem', fontWeight: '900', color: '#0F172A' }}>{pressureVal}</span>
                <span style={{ fontSize: '0.75rem', fontWeight: '700', color: '#64748B', marginLeft: '0.15rem' }}>hPa</span>
              </div>
              <span style={{ fontSize: '0.72rem', color: '#475569', fontWeight: '700' }}>{inversionRisk}</span>
            </div>

            {/* WIND VECTOR */}
            <div style={{ backgroundColor: '#F8FAFC', padding: '1rem', borderRadius: '0.85rem', border: '1px solid #F1F5F9', display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <div style={{ backgroundColor: '#CCFBF1', padding: '0.35rem', borderRadius: '0.4rem', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <Wind size={15} color="#0D9488" />
                </div>
                <span style={{ fontSize: '0.7rem', fontWeight: '800', color: '#64748B', textTransform: 'uppercase' }}>WIND VECTOR</span>
              </div>
              <div style={{ margin: '0.5rem 0' }}>
                <span style={{ fontSize: '1.5rem', fontWeight: '900', color: '#0F172A' }}>{windVal}</span>
                <span style={{ fontSize: '0.75rem', fontWeight: '700', color: '#64748B', marginLeft: '0.15rem' }}>km/h</span>
              </div>
              <span style={{ fontSize: '0.72rem', color: '#0284C7', fontWeight: '700' }}>{windDir}</span>
            </div>

          </div>
        </div>

      </div>

      {/* SECTION 2: Mathematical Persistence Lift Benchmark Banner (Full Width Below) */}
      <div style={{
        backgroundColor: '#FFFFFF',
        borderRadius: '1.25rem',
        padding: '1.25rem 1.6rem',
        border: '1px solid #E2E8F0',
        boxShadow: '0 4px 18px -2px rgba(15, 23, 42, 0.04)',
        display: 'flex',
        flexDirection: 'column',
        gap: '1rem',
        position: 'relative',
        overflow: 'hidden'
      }}>
        {/* Top Accent Line - Multi-color Gradient */}
        <div style={{ position: 'absolute', top: 0, left: 0, right: 0, height: '3px', background: 'linear-gradient(90deg, #0D9488 0%, #0284C7 50%, #7C3AED 100%)' }} />

        {/* Header Line - Spacious & Non-Overlapping */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.8rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem' }}>
            <div style={{ backgroundColor: '#EEF2FF', padding: '0.45rem', borderRadius: '0.5rem', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <Trophy size={18} color="#4338CA" />
            </div>
            <div>
              <h4 style={{ margin: 0, fontSize: '0.95rem', fontWeight: '800', color: '#0F172A' }}>
                Mathematical Persistence Lift Benchmark
              </h4>
              <p style={{ margin: '0.15rem 0 0 0', fontSize: '0.76rem', color: '#64748B' }}>
                Error reduction performance evaluated over naive baseline persistence model (t+H)
              </p>
            </div>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.74rem', fontWeight: '800', color: '#0F766E', backgroundColor: '#F0FDFA', padding: '0.35rem 0.75rem', borderRadius: '0.45rem', border: '1px solid #CCFBF1' }}>
            <ShieldCheck size={16} color="#0D9488" />
            <span>AUTO-PROMOTION GATE: ACTIVE</span>
          </div>
        </div>

        {/* 3 Horizon Stat Cards Grid */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1rem' }}>
          
          {/* Day 1 (+24h) - Emerald Teal */}
          <div style={{ backgroundColor: '#F8FAFC', padding: '1rem 1.15rem', borderRadius: '0.85rem', border: '1px solid #E2E8F0', display: 'flex', flexDirection: 'column', gap: '0.35rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span style={{ fontSize: '0.68rem', fontWeight: '800', color: '#0F766E', backgroundColor: '#F0FDFA', padding: '0.2rem 0.5rem', borderRadius: '0.35rem', border: '1px solid #CCFBF1', textTransform: 'uppercase', letterSpacing: '0.03em' }}>
                Day 1 (+24h Horizon)
              </span>
            </div>
            <div style={{ display: 'flex', alignItems: 'baseline', gap: '0.35rem', marginTop: '0.2rem' }}>
              <span style={{ fontSize: '1.5rem', fontWeight: '900', color: isDataLoading ? '#64748B' : '#0F766E', lineHeight: 1 }}>
                {d1Lift}
              </span>
              <span style={{ fontSize: '0.78rem', fontWeight: '800', color: '#0D9488' }}>
                LIFT
              </span>
            </div>
            <div style={{ fontSize: '0.72rem', color: '#64748B', fontWeight: '600' }}>
              RMSE: {d1Rmse}
            </div>
          </div>

          {/* Day 2 (+48h) - Sapphire Blue */}
          <div style={{ backgroundColor: '#F8FAFC', padding: '1rem 1.15rem', borderRadius: '0.85rem', border: '1px solid #E2E8F0', display: 'flex', flexDirection: 'column', gap: '0.35rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span style={{ fontSize: '0.68rem', fontWeight: '800', color: '#0369A1', backgroundColor: '#F0F9FF', padding: '0.2rem 0.5rem', borderRadius: '0.35rem', border: '1px solid #BAE6FD', textTransform: 'uppercase', letterSpacing: '0.03em' }}>
                Day 2 (+48h Horizon)
              </span>
            </div>
            <div style={{ display: 'flex', alignItems: 'baseline', gap: '0.35rem', marginTop: '0.2rem' }}>
              <span style={{ fontSize: '1.5rem', fontWeight: '900', color: isDataLoading ? '#64748B' : '#0369A1', lineHeight: 1 }}>
                {d2Lift}
              </span>
              <span style={{ fontSize: '0.78rem', fontWeight: '800', color: '#0284C7' }}>
                LIFT
              </span>
            </div>
            <div style={{ fontSize: '0.72rem', color: '#64748B', fontWeight: '600' }}>
              RMSE: {d2Rmse}
            </div>
          </div>

          {/* Day 3 (+72h) - Deep Violet / Purple */}
          <div style={{ backgroundColor: '#F8FAFC', padding: '1rem 1.15rem', borderRadius: '0.85rem', border: '1px solid #E2E8F0', display: 'flex', flexDirection: 'column', gap: '0.35rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span style={{ fontSize: '0.68rem', fontWeight: '800', color: '#6D28D9', backgroundColor: '#F5F3FF', padding: '0.2rem 0.5rem', borderRadius: '0.35rem', border: '1px solid #DDD6FE', textTransform: 'uppercase', letterSpacing: '0.03em' }}>
                Day 3 (+72h Horizon)
              </span>
            </div>
            <div style={{ display: 'flex', alignItems: 'baseline', gap: '0.35rem', marginTop: '0.2rem' }}>
              <span style={{ fontSize: '1.5rem', fontWeight: '900', color: isDataLoading ? '#64748B' : '#6D28D9', lineHeight: 1 }}>
                {d3Lift}
              </span>
              <span style={{ fontSize: '0.78rem', fontWeight: '800', color: '#7C3AED' }}>
                LIFT
              </span>
            </div>
            <div style={{ fontSize: '0.72rem', color: '#64748B', fontWeight: '600' }}>
              RMSE: {d3Rmse}
            </div>
          </div>

        </div>

      </div>

    </div>
  );
}
