namespace RiskManager.Api.Models;

public class HealthStatus
{
    public string PodName { get; set; } = string.Empty;
    public string Uptime { get; set; } = string.Empty;
    public long UptimeSeconds { get; set; }
    public DateTime Timestamp { get; set; }
    public bool DatabaseConnected { get; set; }
    public string DatabaseLatency { get; set; } = string.Empty;
    public string Version { get; set; } = "1.0.0";
}
