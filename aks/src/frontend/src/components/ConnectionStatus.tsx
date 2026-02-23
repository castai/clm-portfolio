import type { HealthStatus } from '../types';

interface ConnectionStatusProps {
  health: HealthStatus | null;
  isConnected: boolean;
  latency: number;
  error: string | null;
}

export function ConnectionStatus({ health, isConnected, latency, error }: ConnectionStatusProps) {
  return (
    <footer style={{
      background: '#0d1117',
      borderTop: '1px solid #1e2a3a',
      padding: '10px 32px',
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
      fontSize: '12px',
      fontFamily: 'monospace',
    }}>
      <div style={{ display: 'flex', gap: '24px', alignItems: 'center' }}>
        <span style={{ color: '#8892b0' }}>
          Pod:{' '}
          <span style={{ color: '#58a6ff', fontWeight: 600 }}>
            {health?.podName || '---'}
          </span>
        </span>
        <span style={{ color: '#8892b0' }}>
          Uptime:{' '}
          <span style={{ color: '#fff' }}>
            {health?.uptime || '---'}
          </span>
        </span>
        <span style={{ color: '#8892b0' }}>
          DB Latency:{' '}
          <span style={{ color: '#fff' }}>
            {health?.databaseLatency || '---'}
          </span>
        </span>
      </div>
      <div style={{ display: 'flex', gap: '24px', alignItems: 'center' }}>
        <span style={{ color: '#8892b0' }}>
          API Latency:{' '}
          <span style={{ color: latency < 100 ? '#00d4aa' : latency < 300 ? '#f0c040' : '#e94560' }}>
            {latency}ms
          </span>
        </span>
        <span style={{
          color: isConnected ? '#00d4aa' : '#e94560',
          fontWeight: 600,
        }}>
          {isConnected ? 'CONNECTED' : `DISCONNECTED: ${error}`}
        </span>
        <span style={{ color: '#8892b0' }}>
          v{health?.version || '---'}
        </span>
      </div>
    </footer>
  );
}
