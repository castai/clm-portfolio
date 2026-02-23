namespace RiskManager.Api.Models;

public class RiskMetrics
{
    public decimal VaR95 { get; set; }
    public decimal VaR99 { get; set; }
    public decimal Beta { get; set; }
    public decimal SharpeRatio { get; set; }
    public decimal Volatility { get; set; }
    public decimal MaxDrawdown { get; set; }
    public decimal TotalAum { get; set; }
    public DateTime CalculatedAt { get; set; }
}
