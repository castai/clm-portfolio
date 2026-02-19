import { Doughnut } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  ArcElement,
  Tooltip,
  Legend,
} from 'chart.js';
import type { SectorAllocation } from '../types';

ChartJS.register(ArcElement, Tooltip, Legend);

const COLORS = [
  '#4361ee', '#3a0ca3', '#7209b7', '#f72585', '#4cc9f0',
  '#06d6a0', '#ffd166', '#ef476f', '#118ab2', '#073b4c',
];

interface PortfolioChartProps {
  sectors: SectorAllocation[];
}

export function PortfolioChart({ sectors }: PortfolioChartProps) {
  const data = {
    labels: sectors.map(s => s.sector),
    datasets: [{
      data: sectors.map(s => s.weight),
      backgroundColor: COLORS.slice(0, sectors.length),
      borderColor: '#0d1117',
      borderWidth: 2,
    }],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'right' as const,
        labels: {
          color: '#8892b0',
          font: { size: 11 },
          padding: 12,
        },
      },
      tooltip: {
        callbacks: {
          label: (ctx: any) => {
            const sector = sectors[ctx.dataIndex];
            return `${sector.sector}: ${sector.weight}% ($${(sector.marketValue / 1e6).toFixed(1)}M)`;
          },
        },
      },
    },
  };

  return (
    <div style={{
      background: '#161b22',
      borderRadius: '12px',
      border: '1px solid #1e2a3a',
      padding: '20px',
      height: '100%',
    }}>
      <h3 style={{ color: '#fff', margin: '0 0 16px 0', fontSize: '14px', fontWeight: 600 }}>
        Asset Allocation
      </h3>
      <div style={{ height: '240px' }}>
        <Doughnut data={data} options={options} />
      </div>
    </div>
  );
}
