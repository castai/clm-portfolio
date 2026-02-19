import axios from 'axios';
import type { Position, RiskMetrics, PortfolioSummary, PnLDataPoint, HealthStatus } from '../types';

const MAX_RETRIES = 3;
const BASE_DELAY_MS = 500;
const REQUEST_TIMEOUT_MS = 8000;

const api = axios.create({
  baseURL: '/api',
  timeout: REQUEST_TIMEOUT_MS,
});

// Retry interceptor with exponential backoff
api.interceptors.response.use(
  (response) => response,
  async (error) => {
    const config = error.config;

    // Don't retry if we've exhausted attempts, or it was cancelled, or it's a 4xx (client error)
    if (
      !config ||
      config._retryCount >= MAX_RETRIES ||
      axios.isCancel(error) ||
      (error.response && error.response.status >= 400 && error.response.status < 500)
    ) {
      return Promise.reject(error);
    }

    config._retryCount = (config._retryCount || 0) + 1;

    // Exponential backoff: 500ms, 1000ms, 2000ms
    const delay = BASE_DELAY_MS * Math.pow(2, config._retryCount - 1);
    await new Promise((resolve) => setTimeout(resolve, delay));

    return api(config);
  }
);

export const getHealth = () => api.get<HealthStatus>('/health').then(r => r.data);
export const getPositions = () => api.get<Position[]>('/portfolio/positions').then(r => r.data);
export const getSummary = () => api.get<PortfolioSummary>('/portfolio/summary').then(r => r.data);
export const getRiskMetrics = () => api.get<RiskMetrics>('/risk/metrics').then(r => r.data);
export const getPnL = () => api.get<PnLDataPoint[]>('/risk/pnl').then(r => r.data);
