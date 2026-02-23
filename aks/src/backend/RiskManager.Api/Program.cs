using Microsoft.EntityFrameworkCore;
using RiskManager.Api.Data;
using RiskManager.Api.Services;

var builder = WebApplication.CreateBuilder(args);

// Add services
builder.Services.AddControllers();
builder.Services.AddEndpointsApiExplorer();

// Market simulator - ticks prices every 5s for live feel
builder.Services.AddHostedService<MarketSimulator>();

// Azure SQL connection
var connectionString = Environment.GetEnvironmentVariable("AZURE_SQL_CONNECTION_STRING")
    ?? builder.Configuration.GetConnectionString("DefaultConnection");

builder.Services.AddDbContext<AppDbContext>(options =>
    options.UseSqlServer(connectionString, sqlOptions =>
    {
        sqlOptions.EnableRetryOnFailure(
            maxRetryCount: 5,
            maxRetryDelay: TimeSpan.FromSeconds(10),
            errorNumbersToAdd: null);
        sqlOptions.CommandTimeout(15); // 15s per query - fail fast, let retry handle it
    }));

// CORS - allow all for demo
builder.Services.AddCors(options =>
{
    options.AddDefaultPolicy(policy =>
    {
        policy.AllowAnyOrigin()
              .AllowAnyMethod()
              .AllowAnyHeader();
    });
});

var app = builder.Build();

// Initialize database
using (var scope = app.Services.CreateScope())
{
    var context = scope.ServiceProvider.GetRequiredService<AppDbContext>();
    try
    {
        await DbInitializer.InitializeAsync(context);
        Console.WriteLine("Database initialized successfully.");
    }
    catch (Exception ex)
    {
        Console.WriteLine($"Database initialization error: {ex.Message}");
        Console.WriteLine("The application will start but database features may not work.");
    }
}

app.UseCors();
app.MapControllers();

Console.WriteLine($"Risk Manager API starting on pod: {Environment.MachineName}");

app.Run();
