using System.Diagnostics;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using RiskManager.Api.Data;
using RiskManager.Api.Models;

namespace RiskManager.Api.Controllers;

[ApiController]
[Route("api/[controller]")]
public class HealthController : ControllerBase
{
    private static readonly DateTime StartTime = DateTime.UtcNow;
    private readonly AppDbContext _context;

    public HealthController(AppDbContext context)
    {
        _context = context;
    }

    [HttpGet]
    public async Task<ActionResult<HealthStatus>> GetHealth(CancellationToken ct)
    {
        var uptime = DateTime.UtcNow - StartTime;
        var dbConnected = false;
        var dbLatency = "N/A";

        try
        {
            // 5s timeout for the DB ping - don't let a slow DB hang the health check
            using var cts = CancellationTokenSource.CreateLinkedTokenSource(ct);
            cts.CancelAfter(TimeSpan.FromSeconds(5));

            var sw = Stopwatch.StartNew();
            await _context.Database.ExecuteSqlRawAsync("SELECT 1", cts.Token);
            sw.Stop();
            dbConnected = true;
            dbLatency = $"{sw.ElapsedMilliseconds}ms";
        }
        catch
        {
            dbConnected = false;
            dbLatency = "timeout";
        }

        return Ok(new HealthStatus
        {
            PodName = Environment.MachineName,
            Uptime = FormatUptime(uptime),
            UptimeSeconds = (long)uptime.TotalSeconds,
            Timestamp = DateTime.UtcNow,
            DatabaseConnected = dbConnected,
            DatabaseLatency = dbLatency,
            Version = "1.0.0"
        });
    }

    private static string FormatUptime(TimeSpan ts)
    {
        if (ts.TotalHours >= 1)
            return $"{(int)ts.TotalHours}h {ts.Minutes}m {ts.Seconds}s";
        if (ts.TotalMinutes >= 1)
            return $"{ts.Minutes}m {ts.Seconds}s";
        return $"{ts.Seconds}s";
    }
}
