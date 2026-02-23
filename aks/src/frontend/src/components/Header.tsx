import type { HealthStatus } from '../types';

interface HeaderProps {
  health: HealthStatus | null;
  isConnected: boolean;
}

export function Header({ health, isConnected }: HeaderProps) {
  return (
    <header style={{
      background: 'linear-gradient(135deg, #1a1a2e 0%, #16213e 100%)',
      padding: '16px 32px',
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
      borderBottom: '2px solid #0f3460',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
        <div style={{
          width: '40px',
          height: '40px',
          background: 'linear-gradient(135deg, #e94560, #0f3460)',
          borderRadius: '8px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          fontSize: '20px',
          fontWeight: 'bold',
          color: '#fff',
        }}>
          RM
        </div>
        <div>
          <h1 style={{ margin: 0, fontSize: '20px', color: '#fff', fontWeight: 700 }}>
            Risk Manager
          </h1>
          <span style={{ fontSize: '12px', color: '#8892b0' }}>
            Portfolio Analytics Platform
          </span>
        </div>
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: '24px' }}>
        {health?.databaseConnected && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <div style={{
              width: '8px', height: '8px', borderRadius: '50%',
              background: '#00d4aa',
              boxShadow: '0 0 6px #00d4aa',
            }} />
            <span style={{ color: '#8892b0', fontSize: '12px' }}>Azure SQL</span>
          </div>
        )}
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <div style={{
            width: '10px',
            height: '10px',
            borderRadius: '50%',
            background: isConnected ? '#00d4aa' : '#e94560',
            boxShadow: isConnected ? '0 0 8px #00d4aa' : '0 0 8px #e94560',
            animation: 'pulse 2s infinite',
          }} />
          <span style={{ color: isConnected ? '#00d4aa' : '#e94560', fontSize: '13px', fontWeight: 600 }}>
            {isConnected ? 'LIVE' : 'RECONNECTING'}
          </span>
        </div>
      </div>
    </header>
  );
}
