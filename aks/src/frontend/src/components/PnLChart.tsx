import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Tooltip,
  Legend,
  Filler,
} from 'chart.js';
import type { PnLDataPoint } from '../types';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Tooltip, Legend, Filler);

interface PnLChartProps {
  pnlData: PnLDataPoint[];
}

export function PnLChart({ pnlData }: PnLChartProps) {
  const labels = pnlData.map(p => {
    const d = new Date(p.date);
    return `${d.getMonth() + 1}/${d.getDate()}`;
  });

  const data = {
    labels,
    datasets: [
      {
        label: 'Portfolio Value ($M)',
        data: pnlData.map(p => p.portfolioValue / 1e6),
        borderColor: '#4361ee',
        backgroundColor: 'rgba(67, 97, 238, 0.1)',
        fill: true,
        tension: 0.3,
        pointRadius: 2,
        pointHoverRadius: 5,
      },
      {
        label: 'Cumulative P&L ($K)',
        data: pnlData.map(p => p.cumulativePnL / 1e3),
        borderColor: '#00d4aa',
        backgroundColor: 'transparent',
        tension: 0.3,
        pointRadius: 2,
        pointHoverRadius: 5,
        yAxisID: 'y1',
      },
    ],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: {
      mode: 'index' as const,
      intersect: false,
    },
    plugins: {
      legend: {
        labels: { color: '#8892b0', font: { size: 11 } },
      },
    },
    scales: {
      x: {
        ticks: { color: '#8892b0', font: { size: 10 } },
        grid: { color: 'rgba(255,255,255,0.05)' },
      },
      y: {
        position: 'left' as const,
        ticks: {
          color: '#8892b0',
          font: { size: 10 },
          callback: (v: any) => `$${v}M`,
        },
        grid: { color: 'rgba(255,255,255,0.05)' },
      },
      y1: {
        position: 'right' as const,
        ticks: {
          color: '#00d4aa',
          font: { size: 10 },
          callback: (v: any) => `$${v}K`,
        },
        grid: { drawOnChartArea: false },
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
        P&L Performance (30 Days)
      </h3>
      <div style={{ height: '240px' }}>
        <Line data={data} options={options} />
      </div>
    </div>
  );
}
