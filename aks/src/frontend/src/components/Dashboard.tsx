import { usePolling } from '../hooks/usePolling';
import { getHealth, getPositions, getSummary, getRiskMetrics, getPnL } from '../services/api';
import { Header } from './Header';
import { ConnectionStatus } from './ConnectionStatus';
import { PortfolioChart } from './PortfolioChart';
import { PnLChart } from './PnLChart';
import { RiskGauge } from './RiskGauge';
import { PositionsTable } from './PositionsTable';

export function Dashboard() {
  const health = usePolling(getHealth, 2000);
  const positions = usePolling(getPositions, 2000);
  const summary = usePolling(getSummary, 2000);
  const risk = usePolling(getRiskMetrics, 2000);
  const pnl = usePolling(getPnL, 2000);

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      height: '100vh',
      background: '#0d1117',
      color: '#c9d1d9',
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
    }}>
      <Header health={health.data} isConnected={health.isConnected} />

      <main style={{
        flex: 1,
        overflow: 'auto',
        padding: '24px',
        display: 'flex',
        flexDirection: 'column',
        gap: '20px',
      }}>
        {/* Charts row */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 2fr', gap: '20px', minHeight: '300px' }}>
          <PortfolioChart sectors={summary.data?.sectorAllocations || []} />
          <PnLChart pnlData={pnl.data || []} />
        </div>

        {/* Risk metrics */}
        <RiskGauge metrics={risk.data} />

        {/* Positions table */}
        <PositionsTable positions={positions.data || []} />
      </main>

      <ConnectionStatus
        health={health.data}
        isConnected={health.isConnected}
        latency={health.latency}
        error={health.error}
      />
    </div>
  );
}
