import { useState, useEffect, useRef, useCallback } from 'react';

export function usePolling<T>(
  fetcher: () => Promise<T>,
  intervalMs: number = 2000
): { data: T | null; error: string | null; latency: number; isConnected: boolean } {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [latency, setLatency] = useState<number>(0);
  const [isConnected, setIsConnected] = useState<boolean>(true);

  const intervalRef = useRef<number | null>(null);
  const fetcherRef = useRef(fetcher);
  const inFlightRef = useRef(false);

  fetcherRef.current = fetcher;

  const poll = useCallback(async () => {
    // Don't stack requests - skip if previous is still in-flight
    if (inFlightRef.current) return;

    inFlightRef.current = true;
    const start = performance.now();

    try {
      const result = await fetcherRef.current();
      const elapsed = Math.round(performance.now() - start);
      setData(result);
      setLatency(elapsed);
      setError(null);
      setIsConnected(true);
    } catch (err: any) {
      const elapsed = Math.round(performance.now() - start);
      setLatency(elapsed);

      // Distinguish timeout vs network error for the UI
      const isTimeout = err.code === 'ECONNABORTED' || err.message?.includes('timeout');
      const msg = isTimeout
        ? 'Request timeout'
        : err.response
          ? `Server error (${err.response.status})`
          : err.message || 'Connection failed';

      setError(msg);
      setIsConnected(false);
      // Keep last known good data on screen - don't clear it
    } finally {
      inFlightRef.current = false;
    }
  }, []);

  useEffect(() => {
    poll(); // initial fetch
    intervalRef.current = window.setInterval(poll, intervalMs);
    return () => {
      if (intervalRef.current) window.clearInterval(intervalRef.current);
    };
  }, [poll, intervalMs]);

  return { data, error, latency, isConnected };
}
