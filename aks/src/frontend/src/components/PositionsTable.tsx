import type { Position } from '../types';

interface PositionsTableProps {
  positions: Position[];
}

function formatCurrency(value: number): string {
  if (Math.abs(value) >= 1e6) return `$${(value / 1e6).toFixed(2)}M`;
  if (Math.abs(value) >= 1e3) return `$${(value / 1e3).toFixed(0)}K`;
  return `$${value.toFixed(0)}`;
}

export function PositionsTable({ positions }: PositionsTableProps) {
  const sorted = [...positions].sort((a, b) => b.marketValue - a.marketValue);

  return (
    <div style={{
      background: '#161b22',
      borderRadius: '12px',
      border: '1px solid #1e2a3a',
      padding: '20px',
      overflow: 'auto',
    }}>
      <h3 style={{ color: '#fff', margin: '0 0 16px 0', fontSize: '14px', fontWeight: 600 }}>
        Portfolio Positions ({positions.length})
      </h3>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '13px' }}>
        <thead>
          <tr style={{ borderBottom: '1px solid #1e2a3a' }}>
            {['Ticker', 'Company', 'Sector', 'Shares', 'Price', 'Cost Basis', 'Market Value', 'P&L', 'Change %'].map(h => (
              <th key={h} style={{
                textAlign: h === 'Ticker' || h === 'Company' || h === 'Sector' ? 'left' : 'right',
                padding: '8px 12px',
                color: '#8892b0',
                fontWeight: 600,
                fontSize: '11px',
                textTransform: 'uppercase',
                letterSpacing: '0.5px',
              }}>
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sorted.map((pos) => (
            <tr key={pos.id} style={{ borderBottom: '1px solid rgba(30,42,58,0.5)' }}>
              <td style={{ padding: '10px 12px', color: '#58a6ff', fontWeight: 600, fontFamily: 'monospace' }}>
                {pos.ticker}
              </td>
              <td style={{ padding: '10px 12px', color: '#c9d1d9' }}>
                {pos.companyName}
              </td>
              <td style={{ padding: '10px 12px', color: '#8892b0' }}>
                {pos.sector}
              </td>
              <td style={{ padding: '10px 12px', color: '#c9d1d9', textAlign: 'right', fontFamily: 'monospace' }}>
                {pos.shares.toLocaleString()}
              </td>
              <td style={{ padding: '10px 12px', color: '#c9d1d9', textAlign: 'right', fontFamily: 'monospace' }}>
                ${pos.price.toFixed(2)}
              </td>
              <td style={{ padding: '10px 12px', color: '#8892b0', textAlign: 'right', fontFamily: 'monospace' }}>
                ${pos.costBasis.toFixed(2)}
              </td>
              <td style={{ padding: '10px 12px', color: '#fff', textAlign: 'right', fontWeight: 600, fontFamily: 'monospace' }}>
                {formatCurrency(pos.marketValue)}
              </td>
              <td style={{
                padding: '10px 12px',
                textAlign: 'right',
                fontFamily: 'monospace',
                color: pos.unrealizedPnL >= 0 ? '#00d4aa' : '#e94560',
              }}>
                {pos.unrealizedPnL >= 0 ? '+' : ''}{formatCurrency(pos.unrealizedPnL)}
              </td>
              <td style={{
                padding: '10px 12px',
                textAlign: 'right',
                fontFamily: 'monospace',
                fontWeight: 600,
                color: pos.changePercent >= 0 ? '#00d4aa' : '#e94560',
              }}>
                {pos.changePercent >= 0 ? '+' : ''}{pos.changePercent}%
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
