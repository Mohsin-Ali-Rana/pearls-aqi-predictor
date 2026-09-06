import React, { useState } from 'react';
import { Cpu, Radio, Layers, Mail, CheckCircle2, AlertCircle, RefreshCw } from 'lucide-react';

export function EmailAlertDispatcher() {
  const [mode, setMode] = useState('subscribe'); // 'subscribe' | 'unsubscribe'
  const [email, setEmail] = useState('');
  const [threshold, setThreshold] = useState(100);
  const [frequency, setFrequency] = useState('6h');
  const [submitting, setSubmitting] = useState(false);
  const [isSubscribed, setIsSubscribed] = useState(false);
  const [unsubscribedEmail, setUnsubscribedEmail] = useState('');
  const [subscribedEmail, setSubscribedEmail] = useState('');
  const [subscribedMeta, setSubscribedMeta] = useState({ threshold: 100, frequency: '6h' });
  const [errorMsg, setErrorMsg] = useState('');

  const handleSubscribe = async (e) => {
    e.preventDefault();
    setErrorMsg('');
    const cleanedEmail = email.trim().toLowerCase();

    if (!cleanedEmail || !cleanedEmail.includes('@') || !cleanedEmail.includes('.')) {
      setErrorMsg('Please enter a valid observer email address (e.g. user@domain.com).');
      return;
    }

    setSubmitting(true);

    try {
      const res = await fetch('/api/subscribe', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          email: cleanedEmail,
          threshold: Number(threshold),
          frequency: frequency
        })
      });
      const json = await res.json();
      if (res.ok) {
        setIsSubscribed(true);
        setSubscribedEmail(cleanedEmail);
        setSubscribedMeta({ threshold: Number(threshold), frequency });
        setEmail('');
        setErrorMsg('');
      } else {
        setErrorMsg(json.detail || 'Subscription service failed. Please try again.');
      }
    } catch (err) {
      setErrorMsg('Network error connecting to dispatch server. Please try again.');
    } finally {
      setSubmitting(false);
    }
  };

  const handleUnsubscribe = async (e) => {
    e.preventDefault();
    setErrorMsg('');
    const cleanedEmail = email.trim().toLowerCase();

    if (!cleanedEmail || !cleanedEmail.includes('@') || !cleanedEmail.includes('.')) {
      setErrorMsg('Please enter a valid observer email address.');
      return;
    }

    setSubmitting(true);

    try {
      const res = await fetch('/api/unsubscribe', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email: cleanedEmail })
      });
      const json = await res.json();
      if (res.ok) {
        setUnsubscribedEmail(cleanedEmail);
        setEmail('');
        setErrorMsg('');
      } else {
        setErrorMsg(json.detail || 'Unsubscription failed. Please verify the email address.');
      }
    } catch (err) {
      setErrorMsg('Network error connecting to server. Please try again.');
    } finally {
      setSubmitting(false);
    }
  };

  const getFreqLabel = (f) => {
    if (f === '24h' || f === 'daily') return 'Max 1 Alert per Day';
    if (f === '6h') return 'Max 1 Alert / 6 Hours';
    return 'Max 1 Alert / Hour';
  };

  if (isSubscribed) {
    return (
      <div style={{
        backgroundColor: '#ECFDF5',
        borderRadius: '1.25rem',
        padding: '1.25rem 1.6rem',
        border: '1.5px solid #A7F3D0',
        boxShadow: '0 4px 20px -2px rgba(16, 185, 129, 0.12)',
        display: 'flex',
        justify: 'space-between',
        alignItems: 'center',
        flexWrap: 'wrap',
        gap: '1rem',
        transition: 'all 0.3s ease'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.9rem' }}>
          <div style={{ width: '42px', height: '42px', borderRadius: '50%', backgroundColor: '#D1FAE5', border: '1px solid #6EE7B7', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
            <CheckCircle2 size={22} color="#059669" />
          </div>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
              <h4 style={{ margin: 0, fontSize: '1.02rem', fontWeight: '800', color: '#065F46', letterSpacing: '-0.015em' }}>
                Automated Alert Dispatcher Active
              </h4>
              <span style={{ backgroundColor: '#059669', color: '#FFFFFF', fontSize: '0.65rem', fontWeight: '900', padding: '0.15rem 0.55rem', borderRadius: '9999px', letterSpacing: '0.04em', textTransform: 'uppercase' }}>
                SUBSCRIBED
              </span>
            </div>
            <p style={{ margin: '0.25rem 0 0 0', fontSize: '0.8rem', color: '#047857', fontWeight: '600' }}>
              SMTP Notifications enabled for <span style={{ fontWeight: '800', textDecoration: 'underline' }}>{subscribedEmail}</span> &bull; Trigger: <span style={{ fontWeight: '800' }}>AQI &gt; {subscribedMeta.threshold}</span> &bull; Cooldown: <span style={{ fontWeight: '800' }}>{getFreqLabel(subscribedMeta.frequency)}</span>
            </p>
          </div>
        </div>

        <button
          onClick={() => setIsSubscribed(false)}
          style={{
            backgroundColor: '#FFFFFF',
            color: '#047857',
            border: '1px solid #A7F3D0',
            borderRadius: '0.7rem',
            padding: '0.5rem 1rem',
            fontSize: '0.8rem',
            fontWeight: '800',
            cursor: 'pointer',
            boxShadow: '0 2px 8px rgba(5, 150, 105, 0.08)',
            transition: 'all 0.2s ease',
            whiteSpace: 'nowrap'
          }}
          onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = '#F0FDFA')}
          onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = '#FFFFFF')}
        >
          Change Settings
        </button>
      </div>
    );
  }

  if (unsubscribedEmail) {
    return (
      <div style={{
        backgroundColor: '#FFF1F2',
        borderRadius: '1.25rem',
        padding: '1.25rem 1.6rem',
        border: '1.5px solid #FECDD3',
        boxShadow: '0 4px 20px -2px rgba(225, 29, 72, 0.08)',
        display: 'flex',
        justify: 'space-between',
        alignItems: 'center',
        flexWrap: 'wrap',
        gap: '1rem'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.9rem' }}>
          <div style={{ width: '42px', height: '42px', borderRadius: '50%', backgroundColor: '#FFE4E6', border: '1px solid #FDA4AF', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
            <CheckCircle2 size={22} color="#E11D48" />
          </div>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
              <h4 style={{ margin: 0, fontSize: '1.02rem', fontWeight: '800', color: '#9F1239', letterSpacing: '-0.015em' }}>
                Unsubscribed Successfully
              </h4>
              <span style={{ backgroundColor: '#E11D48', color: '#FFFFFF', fontSize: '0.65rem', fontWeight: '900', padding: '0.15rem 0.55rem', borderRadius: '9999px', letterSpacing: '0.04em', textTransform: 'uppercase' }}>
                REMOVED
              </span>
            </div>
            <p style={{ margin: '0.25rem 0 0 0', fontSize: '0.8rem', color: '#BE123C', fontWeight: '600' }}>
              <span style={{ fontWeight: '800', textDecoration: 'underline' }}>{unsubscribedEmail}</span> has been removed from all automated AQI hazard dispatches.
            </p>
          </div>
        </div>

        <button
          onClick={() => { setUnsubscribedEmail(''); setMode('subscribe'); }}
          style={{
            backgroundColor: '#FFFFFF',
            color: '#BE123C',
            border: '1px solid #FECDD3',
            borderRadius: '0.7rem',
            padding: '0.5rem 1rem',
            fontSize: '0.8rem',
            fontWeight: '800',
            cursor: 'pointer',
            boxShadow: '0 2px 8px rgba(225, 29, 72, 0.08)',
            whiteSpace: 'nowrap'
          }}
        >
          Resubscribe / Manage Alerts
        </button>
      </div>
    );
  }

  return (
    <div style={{
      backgroundColor: '#FFFFFF',
      borderRadius: '1.25rem',
      padding: '1.3rem 1.6rem',
      border: '1px solid #E2E8F0',
      boxShadow: '0 4px 20px -2px rgba(15, 23, 42, 0.04)',
      display: 'flex',
      flexDirection: 'column',
      gap: '0.8rem',
      color: '#0F172A'
    }}>
      <div style={{
        display: 'flex',
        justify: 'space-between',
        alignItems: 'center',
        flexWrap: 'wrap',
        gap: '1.2rem'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.85rem' }}>
          <div style={{ width: '40px', height: '40px', borderRadius: '0.75rem', backgroundColor: '#F0FDFA', border: '1px solid #CCFBF1', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
            <Mail size={19} color="#0D9488" />
          </div>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
              <h4 style={{ margin: 0, fontSize: '1.02rem', fontWeight: '800', color: '#0F172A', letterSpacing: '-0.015em' }}>
                Automated Hazardous AQI Alert Dispatcher
              </h4>

              {/* Subscribe / Unsubscribe Mode Switcher */}
              <div style={{ display: 'flex', gap: '0.25rem', backgroundColor: '#F1F5F9', padding: '0.2rem', borderRadius: '0.55rem' }}>
                <button
                  type="button"
                  onClick={() => { setMode('subscribe'); setErrorMsg(''); }}
                  style={{
                    backgroundColor: mode === 'subscribe' ? '#FFFFFF' : 'transparent',
                    color: mode === 'subscribe' ? '#0D9488' : '#64748B',
                    border: mode === 'subscribe' ? '1px solid #CBD5E1' : '1px solid transparent',
                    borderRadius: '0.45rem',
                    padding: '0.2rem 0.6rem',
                    fontSize: '0.72rem',
                    fontWeight: '800',
                    cursor: 'pointer',
                    boxShadow: mode === 'subscribe' ? '0 1px 4px rgba(15,23,42,0.06)' : 'none'
                  }}
                >
                  Subscribe
                </button>
                <button
                  type="button"
                  onClick={() => { setMode('unsubscribe'); setErrorMsg(''); }}
                  style={{
                    backgroundColor: mode === 'unsubscribe' ? '#FFFFFF' : 'transparent',
                    color: mode === 'unsubscribe' ? '#E11D48' : '#64748B',
                    border: mode === 'unsubscribe' ? '1px solid #CBD5E1' : '1px solid transparent',
                    borderRadius: '0.45rem',
                    padding: '0.2rem 0.6rem',
                    fontSize: '0.72rem',
                    fontWeight: '800',
                    cursor: 'pointer',
                    boxShadow: mode === 'unsubscribe' ? '0 1px 4px rgba(15,23,42,0.06)' : 'none'
                  }}
                >
                  Unsubscribe
                </button>
              </div>
            </div>
            <p style={{ margin: '0.2rem 0 0 0', fontSize: '0.78rem', color: '#64748B', fontWeight: '500' }}>
              {mode === 'subscribe'
                ? 'Real-time SMTP notification dispatch with customizable threshold & cooldown rules'
                : 'Remove registered email address from automated AQI threshold dispatches'}
            </p>
          </div>
        </div>

        {mode === 'subscribe' ? (
          <form onSubmit={handleSubscribe} style={{ display: 'flex', gap: '0.5rem', alignItems: 'center', flexWrap: 'wrap' }}>
            <input
              type="email"
              value={email}
              onChange={(e) => {
                setEmail(e.target.value);
                if (errorMsg) setErrorMsg('');
              }}
              placeholder="Enter observer email address..."
              style={{
                backgroundColor: '#F8FAFC',
                border: errorMsg ? '1px solid #FCA5A5' : '1px solid #CBD5E1',
                borderRadius: '0.7rem',
                padding: '0.6rem 0.9rem',
                fontSize: '0.82rem',
                color: '#0F172A',
                outline: 'none',
                minWidth: '220px',
                fontWeight: '600'
              }}
            />

            {/* Threshold Selection Dropdown */}
            <select
              value={threshold}
              onChange={(e) => setThreshold(Number(e.target.value))}
              style={{
                backgroundColor: '#F8FAFC',
                border: '1px solid #CBD5E1',
                borderRadius: '0.7rem',
                padding: '0.6rem 0.75rem',
                fontSize: '0.8rem',
                fontWeight: '700',
                color: '#0F172A',
                outline: 'none',
                cursor: 'pointer'
              }}
            >
              <option value={50}>Trigger: AQI &gt; 50 (WHO Safety Limit)</option>
              <option value={100}>Trigger: AQI &gt; 100 (Moderate Caution)</option>
              <option value={150}>Trigger: AQI &gt; 150 (Unhealthy)</option>
              <option value={200}>Trigger: AQI &gt; 200 (Hazardous)</option>
            </select>

            {/* Cooldown Frequency Dropdown (To prevent repeated spam) */}
            <select
              value={frequency}
              onChange={(e) => setFrequency(e.target.value)}
              style={{
                backgroundColor: '#F8FAFC',
                border: '1px solid #CBD5E1',
                borderRadius: '0.7rem',
                padding: '0.6rem 0.75rem',
                fontSize: '0.8rem',
                fontWeight: '700',
                color: '#0F172A',
                outline: 'none',
                cursor: 'pointer'
              }}
            >
              <option value="6h">Freq: Max 1 / 6 Hours</option>
              <option value="24h">Freq: Max 1 / Day</option>
              <option value="1h">Freq: Max 1 / Hour</option>
            </select>

            <button
              type="submit"
              disabled={submitting}
              style={{
                backgroundColor: '#0D9488',
                color: '#FFFFFF',
                border: 'none',
                borderRadius: '0.7rem',
                padding: '0.6rem 1.15rem',
                fontSize: '0.82rem',
                fontWeight: '800',
                cursor: submitting ? 'not-allowed' : 'pointer',
                boxShadow: '0 4px 14px rgba(13, 148, 136, 0.28)',
                transition: 'all 0.2s ease',
                whiteSpace: 'nowrap',
                display: 'inline-flex',
                alignItems: 'center',
                gap: '0.4rem'
              }}
            >
              {submitting ? (
                <>
                  <RefreshCw size={14} className="animate-spin" />
                  <span>Subscribing...</span>
                </>
              ) : (
                'Subscribe Alerts'
              )}
            </button>
          </form>
        ) : (
          <form onSubmit={handleUnsubscribe} style={{ display: 'flex', gap: '0.5rem', alignItems: 'center', flexWrap: 'wrap' }}>
            <input
              type="email"
              value={email}
              onChange={(e) => {
                setEmail(e.target.value);
                if (errorMsg) setErrorMsg('');
              }}
              placeholder="Enter registered email address..."
              style={{
                backgroundColor: '#F8FAFC',
                border: errorMsg ? '1px solid #FCA5A5' : '1px solid #CBD5E1',
                borderRadius: '0.7rem',
                padding: '0.6rem 0.9rem',
                fontSize: '0.82rem',
                color: '#0F172A',
                outline: 'none',
                minWidth: '240px',
                fontWeight: '600'
              }}
            />

            <button
              type="submit"
              disabled={submitting}
              style={{
                backgroundColor: '#E11D48',
                color: '#FFFFFF',
                border: 'none',
                borderRadius: '0.7rem',
                padding: '0.6rem 1.15rem',
                fontSize: '0.82rem',
                fontWeight: '800',
                cursor: submitting ? 'not-allowed' : 'pointer',
                boxShadow: '0 4px 14px rgba(225, 29, 72, 0.28)',
                transition: 'all 0.2s ease',
                whiteSpace: 'nowrap',
                display: 'inline-flex',
                alignItems: 'center',
                gap: '0.4rem'
              }}
            >
              {submitting ? (
                <>
                  <RefreshCw size={14} className="animate-spin" />
                  <span>Unsubscribing...</span>
                </>
              ) : (
                'Unsubscribe Alerts'
              )}
            </button>
          </form>
        )}
      </div>

      {errorMsg && (
        <div style={{
          fontSize: '0.76rem',
          color: '#DC2626',
          backgroundColor: '#FEE2E2',
          border: '1px solid #FCA5A5',
          borderRadius: '0.5rem',
          padding: '0.45rem 0.85rem',
          fontWeight: '700',
          display: 'inline-flex',
          alignItems: 'center',
          gap: '0.4rem',
          alignSelf: 'flex-start'
        }}>
          <AlertCircle size={14} color="#DC2626" />
          <span>{errorMsg}</span>
        </div>
      )}
    </div>
  );
}

export default function MLOpsTelemetryBar({ systemMetrics, modelName, featureStoreStatus }) {
  const completeness = (systemMetrics && systemMetrics.completeness) || '--';
  const accuracy = (systemMetrics && systemMetrics.accuracy) || '--';
  const rawStatus = (systemMetrics && systemMetrics.status) || 'Operational';
  const displayStatus = rawStatus.replace(/System Status:\s*/gi, '');

  const isConnected = featureStoreStatus !== 'Stale';

  return (
    <div style={{
      backgroundColor: '#FFFFFF',
      borderRadius: '0.85rem',
      padding: '0.55rem 1.15rem',
      border: '1px solid #E2E8F0',
      boxShadow: '0 2px 10px rgba(15, 23, 42, 0.02)',
      display: 'flex',
      alignItems: 'center',
      justify: 'space-between',
      flexWrap: 'wrap',
      gap: '0.75rem',
      fontSize: '0.76rem',
      color: '#475569'
    }}>
      {/* Left: Section Icon & Title */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
        <Cpu size={16} color="#0D9488" />
        <span style={{ fontWeight: '800', color: '#0F172A', letterSpacing: '-0.01em' }}>
          MLOps Stream:
        </span>
      </div>

      {/* Center: Sleek Horizontal Micro-Pill Badges */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem', flexWrap: 'wrap' }}>
        
        {/* Completeness Pill */}
        <span style={{ display: 'inline-flex', alignItems: 'center', gap: '0.35rem', backgroundColor: '#F0FDFA', border: '1px solid #CCFBF1', color: '#0F766E', fontWeight: '700', padding: '0.2rem 0.6rem', borderRadius: '0.4rem' }}>
          <span style={{ fontWeight: '800', color: '#0D9488' }}>Data:</span> {completeness}
        </span>

        {/* Residual Confidence Pill */}
        <span style={{ display: 'inline-flex', alignItems: 'center', gap: '0.35rem', backgroundColor: '#F0F9FF', border: '1px solid #BAE6FD', color: '#0369A1', fontWeight: '700', padding: '0.2rem 0.6rem', borderRadius: '0.4rem' }}>
          <span style={{ fontWeight: '800', color: '#0284C7' }}>Residual Conf:</span> {accuracy}
        </span>

        {/* Stream Pipeline Pill */}
        <span style={{ display: 'inline-flex', alignItems: 'center', gap: '0.35rem', backgroundColor: isConnected ? '#ECFDF5' : '#FEF3C7', border: `1px solid ${isConnected ? '#A7F3D0' : '#FDE68A'}`, color: isConnected ? '#047857' : '#D97706', fontWeight: '700', padding: '0.2rem 0.6rem', borderRadius: '0.4rem' }}>
          <Radio size={12} className={isConnected ? 'radar-pulse' : ''} />
          {isConnected ? 'Stream Active · Hopsworks Synchronized' : 'Offline / Local Artifact Mode'}
        </span>

        {/* Active Registry Model Pill */}
        <span style={{ display: 'inline-flex', alignItems: 'center', gap: '0.35rem', backgroundColor: '#F5F3FF', border: '1px solid #DDD6FE', color: '#6D28D9', fontWeight: '700', padding: '0.2rem 0.6rem', borderRadius: '0.4rem' }}>
          <Layers size={12} color="#7C3AED" />
          {modelName || '--'}
        </span>

      </div>

      {/* Right: Operational Pulse Badge */}
      <div style={{ display: 'inline-flex', alignItems: 'center', gap: '0.4rem', color: isConnected ? '#047857' : '#D97706', fontWeight: '800', fontSize: '0.74rem' }}>
        <span className="pulse-dot" style={{ width: '6px', height: '6px', borderRadius: '50%', backgroundColor: isConnected ? '#10B981' : '#F59E0B', flexShrink: 0 }} />
        {isConnected ? displayStatus : 'Offline / Local Fallback Mode'}
      </div>
    </div>
  );
}
