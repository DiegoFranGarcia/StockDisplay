const API_URL = 'http://localhost:8000';
let currentChart = null;
let currentSymbol = null;
let autoRefreshInterval = null;
let refreshTimeout = null;

// ===== STOCK DATA FUNCTIONS =====

async function loadStocks(silent = false) {
    const symbols = document.getElementById('symbolInput').value;
    
    if (!symbols.trim()) {
        showError('Please enter at least one stock symbol');
        return;
    }

    if (!silent) {
        document.getElementById('loadBtn').disabled = true;
        document.getElementById('refreshBtn').disabled = true;
        document.getElementById('loading').style.display = 'block';
        document.getElementById('error').style.display = 'none';
        document.getElementById('stockGrid').innerHTML = '';
        closeChart();
        closePrediction();
    }
    
    try {
        const response = await fetch(`${API_URL}/stocks?symbols=${encodeURIComponent(symbols)}`);
        
        if (!response.ok) {
            throw new Error(`HTTP error ${response.status}`);
        }
        
        const stocks = await response.json();
        
        if (!silent) {
            document.getElementById('loading').style.display = 'none';
        }
        
        if (!Array.isArray(stocks)) {
            throw new Error('Invalid response format');
        }
        
        if (stocks.length === 0) {
            showError('No stocks found. Please check your symbols.');
            return;
        }
        
        displayStocks(stocks);
        updateLastUpdatedTime();
        
        if (silent && currentSymbol && document.getElementById('chartContainer').classList.contains('active')) {
            const period = document.getElementById('periodSelect').value;
            await loadChartData(currentSymbol, period, true);
        }
    } catch (error) {
        if (!silent) {
            document.getElementById('loading').style.display = 'none';
            showError(`Error loading stocks: ${error.message}`);
        }
        console.error('Load stocks error:', error);
    } finally {
        if (!silent) {
            document.getElementById('loadBtn').disabled = false;
            document.getElementById('refreshBtn').disabled = false;
        }
    }
}

function displayStocks(stocks) {
    const grid = document.getElementById('stockGrid');
    
    const cardMap = {};
    Array.from(grid.children).forEach(card => {
        cardMap[card.dataset.symbol] = card;
    });
    
    const updatedSymbols = new Set();
    
    stocks.forEach(stock => {
        if (!stock.symbol || typeof stock.price !== 'number') {
            console.warn('Invalid stock data:', stock);
            return;
        }
        
        updatedSymbols.add(stock.symbol);
        
        if (cardMap[stock.symbol]) {
            updateStockCard(cardMap[stock.symbol], stock);
        } else {
            const card = createStockCard(stock);
            grid.appendChild(card);
        }
    });
    
    Array.from(grid.children).forEach(card => {
        if (!updatedSymbols.has(card.dataset.symbol)) {
            card.remove();
        }
    });
}

function createStockCard(stock) {
    const isPositive = stock.change >= 0;
    const changeClass = isPositive ? 'positive' : 'negative';
    const arrow = isPositive ? '▲' : '▼';
    
    const card = document.createElement('div');
    card.className = 'stock-card';
    card.dataset.symbol = stock.symbol;
    card.onclick = () => showStockDetails(stock.symbol);
    
    card.innerHTML = `
        <div class="stock-symbol">${stock.symbol}</div>
        <div class="stock-price">$${stock.price.toFixed(2)}</div>
        <div class="stock-change ${changeClass}">
            ${arrow} $${Math.abs(stock.change).toFixed(2)} (${stock.change_percent.toFixed(2)}%)
        </div>
        <div class="stock-volume">Volume: ${stock.volume.toLocaleString()}</div>
    `;
    
    return card;
}

function updateStockCard(card, stock) {
    const isPositive = stock.change >= 0;
    const changeClass = isPositive ? 'positive' : 'negative';
    const arrow = isPositive ? '▲' : '▼';
    
    card.classList.add('updating');
    setTimeout(() => card.classList.remove('updating'), 300);
    
    const priceElement = card.querySelector('.stock-price');
    priceElement.textContent = `$${stock.price.toFixed(2)}`;
    priceElement.classList.add('updated');
    setTimeout(() => priceElement.classList.remove('updated'), 500);
    
    card.querySelector('.stock-symbol').textContent = stock.symbol;
    
    const changeElement = card.querySelector('.stock-change');
    changeElement.className = `stock-change ${changeClass}`;
    changeElement.innerHTML = `${arrow} $${Math.abs(stock.change).toFixed(2)} (${stock.change_percent.toFixed(2)}%)`;
    
    card.querySelector('.stock-volume').textContent = `Volume: ${stock.volume.toLocaleString()}`;
}

// ===== CHART FUNCTIONS =====

async function showStockDetails(symbol) {
    currentSymbol = symbol;
    document.getElementById('chartContainer').classList.add('active');
    document.getElementById('periodSelect').value = '1d';
    await loadChartData(symbol, '1d');
    
    // Also show prediction
    await showPrediction(symbol);
}

async function updateChartPeriod() {
    const period = document.getElementById('periodSelect').value;
    if (currentSymbol) {
        await loadChartData(currentSymbol, period);
    }
}

async function loadChartData(symbol, period, silent = false) {
    if (symbol !== currentSymbol) {
        console.log('Chart symbol mismatch, aborting');
        return;
    }
    
    document.getElementById('chartTitle').textContent = `${symbol} - ${getPeriodLabel(period)}`;
    
    try {
        let interval = '1d';
        if (period === '1d') {
            interval = '5m';
        } else if (period === '5d') {
            interval = '15m';
        }
        
        const response = await fetch(
            `${API_URL}/stock/${encodeURIComponent(symbol)}/history?period=${period}&interval=${interval}`
        );
        
        if (!response.ok) {
            throw new Error(`HTTP error ${response.status}`);
        }
        
        const data = await response.json();
        
        if (data.error || !data.history || !Array.isArray(data.history)) {
            throw new Error(data.error || 'Invalid chart data');
        }
        
        if (symbol !== currentSymbol) {
            console.log('Symbol changed during fetch, aborting render');
            return;
        }
        
        const labels = data.history.map(d => d.date);
        const prices = data.history.map(d => d.price);
        
        if (prices.length === 0) {
            if (!silent) {
                showError('No chart data available');
            }
            return;
        }
        
        const firstPrice = prices[0];
        const lastPrice = prices[prices.length - 1];
        const isPositive = lastPrice >= firstPrice;
        
        const chartColor = isPositive ? '#10b981' : '#ef4444';
        const chartFillColor = isPositive ? 'rgba(16, 185, 129, 0.1)' : 'rgba(239, 68, 68, 0.1)';
        
        const minPrice = Math.min(...prices);
        const maxPrice = Math.max(...prices);
        const padding = (maxPrice - minPrice) * 0.1 || 1;
        
        if (currentChart) {
            currentChart.destroy();
        }
        
        const ctx = document.getElementById('stockChart').getContext('2d');
        currentChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Price',
                    data: prices,
                    borderColor: chartColor,
                    backgroundColor: chartFillColor,
                    borderWidth: 2,
                    fill: true,
                    tension: 0.3,
                    pointRadius: period === '1d' ? 0 : 1,
                    pointHoverRadius: 5,
                    pointBackgroundColor: chartColor,
                    pointBorderColor: chartColor
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                        backgroundColor: 'rgba(0, 0, 0, 0.8)',
                        titleColor: '#fff',
                        bodyColor: '#fff',
                        borderColor: chartColor,
                        borderWidth: 1,
                        callbacks: {
                            label: function(context) {
                                return 'Price: $' + context.parsed.y.toFixed(2);
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        grid: {
                            color: '#334155'
                        },
                        ticks: {
                            color: '#94a3b8',
                            maxTicksLimit: period === '1d' ? 10 : 12
                        }
                    },
                    y: {
                        grid: {
                            color: '#334155'
                        },
                        ticks: {
                            color: '#94a3b8',
                            callback: function(value) {
                                return '$' + value.toFixed(2);
                            }
                        },
                        min: minPrice - padding,
                        max: maxPrice + padding
                    }
                },
                interaction: {
                    intersect: false,
                    mode: 'index'
                }
            }
        });
    } catch (error) {
        if (!silent) {
            showError(`Error loading chart: ${error.message}`);
        }
        console.error('Chart load error:', error);
    }
}

function getPeriodLabel(period) {
    const labels = {
        '1d': "Today",
        '5d': 'Last Week',
        '1mo': 'Last Month',
        '3mo': 'Last 3 Months',
        '1y': 'Last Year'
    };
    return labels[period] || period;
}

function closeChart() {
    document.getElementById('chartContainer').classList.remove('active');
    if (currentChart) {
        currentChart.destroy();
        currentChart = null;
    }
}

// ===== ML PREDICTION FUNCTIONS =====

async function showPrediction(symbol) {
    const section = document.getElementById('predictionSection');
    const content = document.getElementById('predictionContent');
    
    section.style.display = 'block';
    content.innerHTML = '<div class="prediction-loading">Getting prediction for ' + symbol + '...</div>';
    
    try {
        // First, check if we have enough historical data
        const historyResponse = await fetch(`${API_URL}/db/stock/${symbol}/history?limit=1000`);
        
        if (!historyResponse.ok) {
            // Need to backfill data first
            content.innerHTML = `
                <div class="prediction-card">
                    <p style="color: #94a3b8; margin-bottom: 15px;">
                        📚 This stock needs historical data before predictions can be made.
                    </p>
                    <p style="color: #64748b; font-size: 14px; margin-bottom: 15px;">
                        The ML model requires at least 1000 days of historical data to calculate features.
                    </p>
                    <button class="prediction-button" onclick="backfillData('${symbol}')">
                        📥 Load Historical Data (5 years)
                    </button>
                </div>
            `;
            return;
        }
        
        const historyData = await historyResponse.json();
        
        if (historyData.length < 1000) {
            content.innerHTML = `
                <div class="prediction-card">
                    <p style="color: #94a3b8; margin-bottom: 10px;">
                        ⚠️ Only ${historyData.length} days of data available. Need 1000+ for predictions.
                    </p>
                    <p style="color: #64748b; font-size: 14px; margin-bottom: 15px;">
                        The ML model uses rolling averages over 2, 5, 60, 250, and 1000 day periods.
                    </p>
                    <button class="prediction-button" onclick="backfillData('${symbol}')">
                        📥 Load More Historical Data
                    </button>
                </div>
            `;
            return;
        }
        
        // Make prediction
        const response = await fetch(`${API_URL}/predict/${symbol}`, {
            method: 'POST'
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Prediction failed');
        }
        
        const prediction = await response.json();
        
        // Display prediction
        const directionClass = prediction.prediction === 'UP' ? 'up' : 'down';
        const arrow = prediction.prediction === 'UP' ? '▲' : '▼';
        
        content.innerHTML = `
            <div class="prediction-card">
                <div class="prediction-header">
                    <div class="prediction-symbol">${prediction.symbol}</div>
                    <div class="prediction-direction ${directionClass}">
                        ${arrow} ${prediction.prediction}
                    </div>
                </div>
                
                <div class="prediction-details">
                    <div class="prediction-stat">
                        <div class="prediction-stat-label">Current Price</div>
                        <div class="prediction-stat-value">$${prediction.current_price}</div>
                    </div>
                    
                    <div class="prediction-stat">
                        <div class="prediction-stat-label">Confidence</div>
                        <div class="prediction-stat-value">${prediction.confidence}%</div>
                        <div class="prediction-confidence">
                            <div class="prediction-confidence-bar" style="width: ${prediction.confidence}%"></div>
                        </div>
                    </div>
                    
                    <div class="prediction-stat">
                        <div class="prediction-stat-label">For Date</div>
                        <div class="prediction-stat-value" style="font-size: 16px;">
                            ${new Date(prediction.prediction_date).toLocaleDateString()}
                        </div>
                    </div>
                    
                    <div class="prediction-stat">
                        <div class="prediction-stat-label">Model</div>
                        <div class="prediction-stat-value" style="font-size: 16px;">
                            ${prediction.model_version}
                        </div>
                    </div>
                </div>
                
                <div class="prediction-note">
                    <strong>ℹ️ How it works:</strong> The model analyzes price ratios and trends over 2, 5, 60, 250, and 1000 day periods. 
                    Predictions require ${prediction.threshold * 100}% confidence to signal UP.
                </div>
            </div>
        `;
        
    } catch (error) {
        content.innerHTML = `
            <div class="prediction-error">
                ❌ Error getting prediction: ${error.message}
            </div>
        `;
    }
}

async function backfillData(symbol) {
    const content = document.getElementById('predictionContent');
    content.innerHTML = `
        <div class="prediction-loading">
            📥 Loading historical data for ${symbol}...<br>
            <span style="font-size: 14px; color: #64748b;">This may take a minute.</span>
        </div>
    `;
    
    try {
        const response = await fetch(`${API_URL}/stock/${symbol}/backfill?years=5`);
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Failed to load historical data');
        }
        
        const result = await response.json();
        
        content.innerHTML = `
            <div class="prediction-card">
                <h3 style="color: #10b981; margin-bottom: 15px;">✓ Data Loaded Successfully</h3>
                <p style="color: #94a3b8; margin-bottom: 10px;">
                    📊 Loaded <strong>${result.records_fetched}</strong> records
                </p>
                <p style="color: #64748b; font-size: 14px; margin-bottom: 15px;">
                    Date range: ${result.date_range.start} to ${result.date_range.end}
                </p>
                <button class="prediction-button" onclick="showPrediction('${symbol}')">
                    🔮 Get Prediction Now
                </button>
            </div>
        `;
    } catch (error) {
        content.innerHTML = `
            <div class="prediction-error">
                ❌ Error loading data: ${error.message}
            </div>
        `;
    }
}

function closePrediction() {
    document.getElementById('predictionSection').style.display = 'none';
}

// ===== UTILITY FUNCTIONS =====

function refreshStocks() {
    if (refreshTimeout) {
        console.log('Refresh already in progress');
        return;
    }
    
    refreshTimeout = setTimeout(() => {
        refreshTimeout = null;
    }, 1000);
    
    loadStocks(false);
}

function showError(message) {
    const errorDiv = document.getElementById('error');
    errorDiv.textContent = message;
    errorDiv.style.display = 'block';
    
    setTimeout(() => {
        errorDiv.style.display = 'none';
    }, 5000);
}

function updateLastUpdatedTime() {
    const now = new Date();
    const timeString = now.toLocaleTimeString();
    document.getElementById('lastUpdated').textContent = `Last updated: ${timeString}`;
}

function startAutoRefresh() {
    if (autoRefreshInterval) {
        clearInterval(autoRefreshInterval);
    }
    
    autoRefreshInterval = setInterval(() => {
        loadStocks(true);
    }, 120000); // 2 minutes
}

function stopAutoRefresh() {
    if (autoRefreshInterval) {
        clearInterval(autoRefreshInterval);
        autoRefreshInterval = null;
    }
}

// ===== INITIALIZATION =====

window.onload = () => {
    loadStocks(false);
    startAutoRefresh();
};

window.addEventListener('beforeunload', () => {
    stopAutoRefresh();
});

document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
        stopAutoRefresh();
    } else {
        startAutoRefresh();
        loadStocks(true);
    }
});