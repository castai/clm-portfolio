export interface Position {
  id: number;
  ticker: string;
  companyName: string;
  sector: string;
  shares: number;
  price: number;
  costBasis: number;
  marketValue: number;
  unrealizedPnL: number;
  changePercent: number;
}

export interface RiskMetrics {
  vaR95: number;
  vaR99: number;
  beta: number;
  sharpeRatio: number;
  volatility: number;
  maxDrawdown: number;
  totalAum: number;
  calculatedAt: string;
}

export interface SectorAllocation {
  sector: string;
  marketValue: number;
  weight: number;
}

export interface TopHolding {
  ticker: string;
  companyName: string;
  marketValue: number;
  weight: number;
}

export interface PortfolioSummary {
  totalAum: number;
  totalPnL: number;
  pnLPercent: number;
  positionCount: number;
  sectorAllocations: SectorAllocation[];
  topHoldings: TopHolding[];
}

export interface PnLDataPoint {
  id: number;
  date: string;
  dailyPnL: number;
  cumulativePnL: number;
  portfolioValue: number;
}

export interface HealthStatus {
  podName: string;
  uptime: string;
  uptimeSeconds: number;
  timestamp: string;
  databaseConnected: boolean;
  databaseLatency: string;
  version: string;
}
