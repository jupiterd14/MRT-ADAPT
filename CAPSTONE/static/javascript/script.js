  
    let chart = null;
    let currentStation = '{{ favorite_station if favorite_station else "North Ave" }}';
    let updateInterval = null;
    let watchId = null;
    let gpsWatchdog = null;
    let gpsAttempts = 0;
    const MAX_GPS_ATTEMPTS = 3;
    
    // Advisory update control
    let lastAdvisoryUpdate = 0;
    const ADVISORY_UPDATE_INTERVAL = 10 * 60 * 1000; // 10 minutes
    let historicalPatterns = {};

    // Station coordinates
    const stationCoords = {
        "North Ave": { lat: 14.6556, lng: 121.0302 },
        "Quezon Ave": { lat: 14.6390, lng: 121.0380 },
        "Kamuning": { lat: 14.6249, lng: 121.0431 },
        "Cubao": { lat: 14.6213, lng: 121.0529 },
        "Santolan": { lat: 14.6135, lng: 121.0630 },
        "Ortigas": { lat: 14.5864, lng: 121.0565 },
        "Shaw Blvd": { lat: 14.5789, lng: 121.0532 },
        "Boni Ave": { lat: 14.5716, lng: 121.0492 },
        "Guadalupe": { lat: 14.5655, lng: 121.0446 },
        "Buendia": { lat: 14.5547, lng: 121.0329 },
        "Ayala Ave": { lat: 14.5497, lng: 121.0305 },
        "Magallanes": { lat: 14.5450, lng: 121.0254 },
        "Taft": { lat: 14.5378, lng: 121.0112 }
    };

    document.addEventListener('DOMContentLoaded', function() {
        startAutomaticGPS();
        loadStationData(currentStation);
        startAutoRefresh();
        checkAlertCount();
        fetchHistoricalPatterns();
    });

    function fetchHistoricalPatterns() {
        fetch('/api/historical-patterns')
            .then(res => res.json())
            .then(data => {
                historicalPatterns = data;
                console.log('✅ Historical patterns loaded');
            })
            .catch(err => {
                console.log('Historical patterns not available');
            });
    }

    function startAutomaticGPS() {
        const gpsStatus = document.getElementById('gpsStatus');
        const manualOverride = document.getElementById('manualOverride');
        
        if (!navigator.geolocation) {
            gpsStatus.innerHTML = '<i class="fas fa-circle" style="color: #EF4444;"></i> GPS not supported';
            manualOverride.classList.add('visible');
            return;
        }

        gpsWatchdog = setTimeout(() => {
            gpsAttempts++;
            if (gpsAttempts >= MAX_GPS_ATTEMPTS) {
                gpsStatus.innerHTML = '<i class="fas fa-circle" style="color: #EF4444;"></i> GPS timeout';
                manualOverride.classList.add('visible');
            }
        }, 10000);

        gpsStatus.innerHTML = '<i class="fas fa-circle" style="color: #F59E0B;"></i> Detecting location...';

        watchId = navigator.geolocation.watchPosition(
            function(position) {
                if (gpsWatchdog) {
                    clearTimeout(gpsWatchdog);
                    gpsWatchdog = null;
                }
                
                gpsAttempts = 0;
                
                gpsStatus.innerHTML = '<i class="fas fa-circle" style="color: #22C55E;"></i> GPS connected';
                manualOverride.classList.remove('visible');
                
                findAndUpdateNearestStation(position.coords.latitude, position.coords.longitude);
            },
            function(error) {
                gpsAttempts++;
                
                let errorMessage = '';
                switch(error.code) {
                    case error.PERMISSION_DENIED:
                        errorMessage = 'Location denied';
                        break;
                    case error.POSITION_UNAVAILABLE:
                        errorMessage = 'Location unavailable';
                        break;
                    case error.TIMEOUT:
                        errorMessage = 'Location timeout';
                        break;
                    default:
                        errorMessage = 'GPS error';
                }
                
                gpsStatus.innerHTML = `<i class="fas fa-circle" style="color: #EF4444;"></i> ${errorMessage}`;
                
                if (gpsAttempts >= MAX_GPS_ATTEMPTS) {
                    manualOverride.classList.add('visible');
                    if (gpsWatchdog) {
                        clearTimeout(gpsWatchdog);
                        gpsWatchdog = null;
                    }
                }
            },
            {
                enableHighAccuracy: true,
                maximumAge: 10000,
                timeout: 15000
            }
        );
    }

    function findAndUpdateNearestStation(lat, lng) {
        let nearestStation = null;
        let minDistance = Infinity;

        for (const [station, coords] of Object.entries(stationCoords)) {
            const distance = calculateDistance(lat, lng, coords.lat, coords.lng);
            if (distance < minDistance) {
                minDistance = distance;
                nearestStation = station;
            }
        }

        if (minDistance < 30 && nearestStation && nearestStation !== currentStation) {
            changeStation(nearestStation);
        } else if (minDistance < 30 && nearestStation) {
            loadStationData(nearestStation, false);
        }
    }

    function calculateDistance(lat1, lon1, lat2, lon2) {
        const R = 6371;
        const dLat = (lat2 - lat1) * Math.PI / 180;
        const dLon = (lon2 - lon1) * Math.PI / 180;
        const a = Math.sin(dLat/2) * Math.sin(dLat/2) +
                 Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) * 
                 Math.sin(dLon/2) * Math.sin(dLon/2);
        const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
        return R * c;
    }

    function startAutoRefresh() {
        if (updateInterval) clearInterval(updateInterval);
        updateInterval = setInterval(() => {
            loadStationData(currentStation, false);
            checkAlertCount();
        }, 300000);
    }

    function changeStation(station) {
        currentStation = station;
        document.getElementById('stationSelector').value = station;
        document.getElementById('forecastStationSelect').value = station;
        
        loadStationData(station);
        
        {% if is_logged_in %}
        fetch('/api/set-favorite', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ station: station })
        });
        {% endif %}
    }

    function changeForecastStation(station) {
        if (station !== currentStation) {
            changeStation(station);
        } else {
            loadStationData(station, true);
        }
    }

    function refreshData() {
        document.body.classList.add('loading');
        Promise.all([
            loadStationData(currentStation, false),
            checkAlertCount()
        ]).finally(() => {
            setTimeout(() => {
                document.body.classList.remove('loading');
            }, 500);
        });
    }

    function calculateTrend(forecast) {
        if (!forecast || forecast.length < 3) return 'stable';
        
        const recent = forecast.slice(0, 3);
        const avg = recent.reduce((a, b) => a + b, 0) / recent.length;
        const first = forecast[0];
        const last = forecast[forecast.length - 1];
        
        if (last > first * 1.15) return 'rising';
        if (last < first * 0.85) return 'falling';
        return 'stable';
    }

    function getFallbackData(station) {
        const now = new Date();
        const hour = now.getHours();
        
        let congestion;
        if (hour >= 7 && hour <= 9) congestion = 85;
        else if (hour >= 17 && hour <= 20) congestion = 90;
        else if (hour >= 10 && hour <= 16) congestion = 60;
        else if (hour >= 5 && hour <= 6) congestion = 30;
        else if (hour >= 21 && hour <= 22) congestion = 40;
        else congestion = 15;
        
        if (station === "Cubao" || station === "Ayala Ave" || station === "North Ave") {
            congestion = Math.min(95, congestion + 10);
        }
        
        return {
            station: station,
            congestion: congestion,
            forecast: [65, 70, 75, 72, 68, 60],
            current: congestion
        };
    }

    function loadStationData(station, showLoading = true) {
    if (showLoading) {
        document.body.classList.add('loading');
    }

    console.log(`📡 Fetching LSTM data for ${station}...`);
    
    return fetch(`/api/predict/${encodeURIComponent(station)}`)
        .then(res => {
            if (!res.ok) throw new Error(`API returned ${res.status}`);
            return res.json();
        })
        .then(data => {
            console.log('✅ LSTM prediction received:', data);
            updateStationCard(data);
            return fetch(`/api/station-forecast/${encodeURIComponent(station)}`);
        })
        .then(res => {
            if (!res.ok) throw new Error(`Forecast API returned ${res.status}`);
            return res.json();
        })
        .then(data => {
            console.log('✅ Forecast data received:', data);
            updateChart(data);
            
            // FORCE update advisory on every load for now
            updateSmartAdvisory(data, station);
            lastAdvisoryUpdate = Date.now();
        })
        .catch(error => {
            console.error('❌ Error loading from API:', error);
            console.log('📊 Using fallback data for', station);
            
            const fallbackData = getFallbackData(station);
            updateStationCard(fallbackData);
            
            const fallbackForecast = {
                forecast: fallbackData.forecast,
                current: fallbackData.congestion
            };
            updateChart(fallbackForecast);
            
            // FORCE update advisory with fallback data
            updateSmartAdvisory(fallbackForecast, station);
            lastAdvisoryUpdate = Date.now();
        })
        .finally(() => {
            if (showLoading) {
                document.body.classList.remove('loading');
            }
        });
}
    function updateStationCard(data) {
        document.getElementById('stationName').textContent = data.station;
        document.getElementById('stationBadge').textContent = data.station;
        document.getElementById('chartTitle').textContent = `${data.station}: NEXT 6 HOURS`;
        
        const stationCard = document.getElementById('stationCard');
        const statusPill = document.getElementById('statusPill');
        const statusText = document.getElementById('statusText');
        
        stationCard.classList.remove('critical-bg', 'busy-bg', 'moderate-bg', 'light-bg');
        
        statusPill.className = 'status-pill';
        if (data.congestion > 80) {
            statusPill.classList.add('critical');
            statusText.textContent = 'SEVERELY CONGESTED';
            stationCard.classList.add('critical-bg');
        } else if (data.congestion > 50) {
            statusPill.classList.add('busy');
            statusText.textContent = 'CONGESTED';
            stationCard.classList.add('busy-bg');
        } else if (data.congestion > 20) {
            statusPill.classList.add('moderate');
            statusText.textContent = 'MODERATE';
            stationCard.classList.add('moderate-bg');
        } else {
            statusPill.classList.add('light');
            statusText.textContent = 'LIGHT';
            stationCard.classList.add('light-bg');
        }
    }

    function calculateWaitTime(congestion, trend, station) {
        let baseWait;
        
        if (congestion <= 20) {
            baseWait = 2 + (congestion / 20) * 2;
        } else if (congestion <= 50) {
            baseWait = 4 + ((congestion - 20) / 30) * 3;
        } else if (congestion <= 80) {
            baseWait = 7 + ((congestion - 50) / 30) * 5;
        } else {
            baseWait = 12 + ((congestion - 80) / 20) * 6;
        }
        
        const stationMultipliers = {
            "North Ave": 1.1, "Cubao": 1.15, "Ayala Ave": 1.1
        };
        baseWait *= (stationMultipliers[station] || 1.0);
        
        if (trend === 'rising') baseWait *= 1.1;
        else if (trend === 'falling') baseWait *= 0.9;
        
        baseWait = Math.max(2, Math.min(20, Math.round(baseWait)));
        
        return {
            min: Math.max(1, baseWait - 2),
            max: baseWait + 2,
            display: `${Math.max(1, baseWait - 2)}-${baseWait + 2}`
        };
    }

   function updateSmartAdvisory(data, station) {
    const waitTimeValue = document.getElementById('waitTimeValue');
    const advisoryText = document.getElementById('advisoryText');
    const bestTimeText = document.getElementById('bestTimeText');
    
    const now = new Date();
    const hour = now.getHours();
    const minute = now.getMinutes();
    
    console.log('📊 updateSmartAdvisory called with data:', data);
    
    // Get current congestion - with better fallback
    let currentCongestion;
    if (data && data.current !== undefined) {
        currentCongestion = data.current;
    } else if (data && data.congestion !== undefined) {
        currentCongestion = data.congestion;
    } else {
        // Try to get from status pill as fallback
        const statusText = document.getElementById('statusText').textContent;
        if (statusText === 'SEVERELY CONGESTED') currentCongestion = 85;
        else if (statusText === 'CONGESTED') currentCongestion = 65;
        else if (statusText === 'MODERATE') currentCongestion = 35;
        else currentCongestion = 15;
    }
    
    console.log('📊 Current congestion:', currentCongestion);
    
    // Get forecast data - ALWAYS ensure we have forecast data
    let forecast = [];
    if (data && data.forecast && Array.isArray(data.forecast) && data.forecast.length > 0) {
        forecast = data.forecast;
    } else if (data && data.predictions && Array.isArray(data.predictions) && data.predictions.length > 0) {
        forecast = data.predictions;
    } else {
        // Generate forecast based on time of day
        const baseCongestion = currentCongestion || 50;
        forecast = [];
        for (let i = 0; i < 6; i++) {
            let variation = Math.floor(Math.random() * 15) - 5;
            let val = Math.min(95, Math.max(15, baseCongestion + variation + (i * 2)));
            forecast.push(val);
        }
    }
    
    console.log('📊 Forecast data:', forecast);
    
    const trend = calculateTrend(forecast);
    const waitTime = calculateWaitTime(currentCongestion, trend, station);
    
    waitTimeValue.innerHTML = `<span>${waitTime.display} min</span>`;
    
    const isTrainRunning = !((hour === 22 && minute >= 30) || hour >= 23 || hour < 4 || (hour === 4 && minute < 30));
    
    if (!isTrainRunning) {
        if (hour >= 23 || hour < 4) {
            advisoryText.innerHTML = `No trains operating. First train at 4:30 AM.`;
        } else if (hour === 22 && minute >= 30) {
            advisoryText.innerHTML = `Last trains departing now. Service ends 10:30 PM.`;
        } else {
            advisoryText.innerHTML = `Train service ended. Resumes at 4:30 AM.`;
        }
        return;
    }
    
    let advisory = '';
    
    if (hour >= 7 && hour <= 9) advisory = `${station} during morning rush. `;
    else if (hour >= 17 && hour <= 20) advisory = `${station} during evening rush. `;
    else if (hour >= 10 && hour <= 16) advisory = `${station} during midday. `;
    else advisory = `${station} now. `;
    
    if (currentCongestion > 70) advisory += `Severe congestion. Wait ${waitTime.display} min. `;
    else if (currentCongestion > 50) advisory += `Heavy traffic. Wait ${waitTime.display} min. `;
    else if (currentCongestion > 30) advisory += `Moderate flow. Wait ${waitTime.display} min. `;
    else advisory += `Light traffic. Trains every ${waitTime.display} min. `;
    
    if (trend === 'rising') advisory += `Crowds building. `;
    else if (trend === 'falling') advisory += `Clearing up. `;
    
    advisoryText.innerHTML = advisory;
    
    // Best time recommendation - ALWAYS try to show something
    try {
        const forecastTimes = [];
        for (let i = 0; i < forecast.length; i++) {
            const forecastHour = (hour + i + 1) % 24;
            
            if (forecastHour >= 5 && forecastHour <= 21) {
                forecastTimes.push({
                    hour: forecastHour,
                    congestion: forecast[i],
                    timeStr: `${((forecastHour % 12) || 12)}:00 ${forecastHour >= 12 ? 'PM' : 'AM'}`,
                    hoursFromNow: i + 1
                });
            }
        }
        
        console.log('📊 Forecast times:', forecastTimes);
        
        if (forecastTimes.length > 0) {
            let bestTime = forecastTimes.reduce((min, t) => t.congestion < min.congestion ? t : min, forecastTimes[0]);
            let worstTime = forecastTimes.reduce((max, t) => t.congestion > max.congestion ? t : max, forecastTimes[0]);
            
            const diff = currentCongestion - bestTime.congestion;
            const timeSaved = Math.round((worstTime.congestion - bestTime.congestion) / 6);
            
            if (currentCongestion <= 30) {
                bestTimeText.innerHTML = `<b>Best time:</b> Travel now — conditions are light (${currentCongestion}% congestion)`;
            }
            else if (diff > 20) {
                bestTimeText.innerHTML = `<b>Best time:</b> ${bestTime.timeStr} — save ~${timeSaved} min (${Math.round(diff)}% lighter)`;
            }
            else if (diff > 10) {
                bestTimeText.innerHTML = `<b>Best time:</b> ${bestTime.timeStr} — slightly better than now`;
            }
            else if (diff < -15) {
                bestTimeText.innerHTML = `<b>Best time:</b> Now is better than ${bestTime.timeStr}`;
            }
            else {
                bestTimeText.innerHTML = `<b>Best time:</b> Travel now — similar to best time (${bestTime.timeStr})`;
            }
        } else {
            if (hour < 5) {
                bestTimeText.innerHTML = `<b>Best time:</b> First train at 5:00 AM`;
            } else if (hour > 21) {
                bestTimeText.innerHTML = `<b>Best time:</b> Last trains now — plan tomorrow`;
            } else {
                bestTimeText.innerHTML = `<b>Best time:</b> Check forecast above`;
            }
        }
    } catch (e) {
        console.error('Error in best time calculation:', e);
        bestTimeText.innerHTML = `<b>Best time:</b> Based on current conditions`;
    }
}

    function updateChart(data) {
        const ctx = document.getElementById('forecastChart').getContext('2d');
        
        if (chart) {
            chart.destroy();
        }
        
        const now = new Date();
        const labels = [];
        for (let i = 0; i < 6; i++) {
            const hour = (now.getHours() + i + 1) % 24;
            const ampm = hour >= 12 ? 'PM' : 'AM';
            const hour12 = hour % 12 || 12;
            labels.push(`${hour12}:00 ${ampm}`);
        }
        
        const forecastData = data.forecast || [];
        
        chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    data: forecastData,
                    borderColor: '#3B82F6',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    borderWidth: 3,
                    pointBackgroundColor: (context) => {
                        const index = context.dataIndex;
                        if (forecastData[index] > 70) return '#EF4444';
                        if (forecastData[index] < 30) return '#22C55E';
                        return '#3B82F6';
                    },
                    pointBorderColor: 'white',
                    pointBorderWidth: 2,
                    pointRadius: 6,
                    pointHoverRadius: 8,
                    tension: 0.2,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    y: { beginAtZero: true, max: 100 }
                }
            }
        });
    }

    function checkAlertCount() {
        fetch(`/api/predict/${encodeURIComponent(currentStation)}`)
            .then(res => res.json())
            .then(data => {
                const badge = document.getElementById('alertBadge');
                const sidebarBadge = document.getElementById('sidebarAlertCount');
                
                if (data.congestion > 70) {
                    badge.textContent = '1';
                    sidebarBadge.textContent = '1';
                    sidebarBadge.style.display = 'block';
                } else {
                    badge.textContent = '0';
                    sidebarBadge.style.display = 'none';
                }
            })
            .catch(() => {});
    }

    window.addEventListener('beforeunload', function() {
        if (updateInterval) clearInterval(updateInterval);
        if (watchId) navigator.geolocation.clearWatch(watchId);
        if (gpsWatchdog) clearTimeout(gpsWatchdog);
    });
