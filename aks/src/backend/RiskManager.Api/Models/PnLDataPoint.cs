namespace RiskManager.Api.Models;

public class PnLDataPoint
{
    public int Id { get; set; }
    public DateTime Date { get; set; }
    public decimal DailyPnL { get; set; }
    public decimal CumulativePnL { get; set; }
    public decimal PortfolioValue { get; set; }
}
