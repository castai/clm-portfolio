import type { RiskMetrics } from '../types';

interface RiskGaugeProps {
  metrics: RiskMetrics | null;
}

function formatCurrency(value: number): string {
  if (Math.abs(value) >= 1e9) return `$${(value / 1e9).toFixed(2)}B`;
  if (Math.abs(value) >= 1e6) return `$${(value / 1e6).toFixed(1)}M`;
  if (Math.abs(value) >= 1e3) return `$${(value / 1e3).toFixed(0)}K`;
  return `$${value.toFixed(0)}`;
}

interface MetricCardProps {
  label: string;
  value: string;
  subLabel?: string;
  color: string;
}

function MetricCard({ label, value, subLabel, color }: MetricCardProps) {
  return (
    <div style={{
      background: '#161b22',
      borderRadius: '12px',
      border: '1px solid #1e2a3a',
      padding: '20px',
      flex: '1 1 0',
      minWidth: '140px',
      textAlign: 'center',
    }}>
      <div style={{ color: '#8892b0', fontSize: '11px', textTransform: 'uppercase', letterSpacing: '1px', marginBottom: '8px' }}>
        {label}
      </div>
      <div style={{ color, fontSize: '24px', fontWeight: 700, fontFamily: 'monospace' }}>
        {value}
      </div>
      {subLabel && (
        <div style={{ color: '#555', fontSize: '10px', marginTop: '4px' }}>
          {subLabel}
        </div>
      )}
    </div>
  );
}

export function RiskGauge({ metrics }: RiskGaugeProps) {
  if (!metrics) return null;

  return (
    <div style={{
      display: 'flex',
      gap: '12px',
      flexWrap: 'wrap',
    }}>
      <MetricCard
        label="Total AUM"
        value={formatCurrency(metrics.totalAum)}
        color="#4361ee"
      />
      <MetricCard
        label="VaR (95%)"
        value={formatCurrency(metrics.vaR95)}
        subLabel="1-day, parametric"
        color="#e94560"
      />
      <MetricCard
        label="VaR (99%)"
        value={formatCurrency(metrics.vaR99)}
        subLabel="1-day, parametric"
        color="#f72585"
      />
      <MetricCard
        label="Beta"
        value={metrics.beta.toFixed(2)}
        subLabel="vs S&P 500"
        color="#ffd166"
      />
      <MetricCard
        label="Sharpe Ratio"
        value={metrics.sharpeRatio.toFixed(2)}
        color="#00d4aa"
      />
      <MetricCard
        label="Volatility"
        value={`${metrics.volatility.toFixed(1)}%`}
        subLabel="annualized"
        color="#4cc9f0"
      />
      <MetricCard
        label="Max Drawdown"
        value={`${metrics.maxDrawdown.toFixed(1)}%`}
        color="#e94560"
      />
    </div>
  );
}
