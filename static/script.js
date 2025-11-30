// Zaytrics - Lapz.io Style Implementation

class ZaytricsDashboard {
    constructor() {
        this.isRunning = false;
        this.currentView = 'live';
        this.peopleCount = 0;
        this.fps = 0;
        this.init();
    }

    init() {
        console.log('Zaytrics Dashboard Initialized - Lapz.io Style');
        this.setupEventListeners();
        this.startDemoData();
        this.animateStats();
    }

    setupEventListeners() {
        // Control button interactions
        document.querySelectorAll('.control-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                document.querySelectorAll('.control-btn').forEach(b => b.classList.remove('active'));
                e.currentTarget.classList.add('active');
            });
        });

        // Primary action buttons - all "Start Monitoring" buttons
        document.querySelectorAll('.btn-primary').forEach(btn => {
            btn.addEventListener('click', () => {
                this.toggleMonitoring();
            });
        });
    }

    async toggleMonitoring() {
        console.log('toggleMonitoring called, current state:', this.isRunning);
        this.isRunning = !this.isRunning;
        const btn = document.querySelector('.btn-primary');
        
        if (this.isRunning) {
            btn.innerHTML = '<span class="btn-icon">⏸</span> Stop Monitoring';
            await this.startRealUpdates();
        } else {
            btn.innerHTML = '<span class="btn-icon">▶</span> Start Monitoring';
            await this.stopRealUpdates();
        }
    }

    startDemoData() {
        // Poll stats from Flask API
        this.statsInterval = setInterval(async () => {
            if (this.isRunning) {
                await this.updateStatsFromAPI();
            }
        }, 1000);
    }

    async updateStatsFromAPI() {
        try {
            const response = await fetch('/api/stats');
            const data = await response.json();
            
            this.peopleCount = data.count || 0;
            this.fps = data.fps || 0;
            
            document.getElementById('peopleCount').textContent = this.peopleCount;
            document.getElementById('fpsCount').textContent = this.fps.toFixed(1);
        } catch (error) {
            console.error('Error fetching stats:', error);
        }
    }

    animateStats() {
        // Animate stat numbers when they change
        const observer = new MutationObserver((mutations) => {
            mutations.forEach((mutation) => {
                if (mutation.type === 'characterData' || mutation.type === 'childList') {
                    const element = mutation.target;
                    if (element.parentElement.classList.contains('stat-value')) {
                        element.parentElement.style.transform = 'scale(1.1)';
                        setTimeout(() => {
                            element.parentElement.style.transform = 'scale(1)';
                        }, 300);
                    }
                }
            });
        });

        const statElements = document.querySelectorAll('.stat-value');
        statElements.forEach(element => {
            observer.observe(element, {
                characterData: true,
                childList: true,
                subtree: true
            });
        });
    }

    async startRealUpdates() {
        console.log('startRealUpdates called');
        try {
            console.log('Fetching /api/start...');
            const response = await fetch('/api/start', { method: 'POST' });
            const data = await response.json();
            console.log('API response:', data);
            
            if (data.status === 'started') {
                // Show video feed
                const placeholder = document.getElementById('videoPlaceholder');
                const videoFeed = document.getElementById('videoFeed');
                
                console.log('Placeholder:', placeholder, 'VideoFeed:', videoFeed);
                
                if (placeholder) placeholder.style.display = 'none';
                if (videoFeed) {
                    videoFeed.style.display = 'block';
                    videoFeed.src = '/video_feed';
                    console.log('Video feed started');
                }
            }
        } catch (error) {
            console.error('Error starting monitoring:', error);
        }
    }

    async stopRealUpdates() {
        console.log('Stopping monitoring...');
        try {
            const response = await fetch('/api/stop', { method: 'POST' });
            const data = await response.json();
            
            if (data.status === 'stopped') {
                // Hide video feed
                const placeholder = document.getElementById('videoPlaceholder');
                const videoFeed = document.getElementById('videoFeed');
                
                if (videoFeed) {
                    videoFeed.style.display = 'none';
                    videoFeed.src = '';
                }
                if (placeholder) placeholder.style.display = 'flex';
            }
        } catch (error) {
            console.error('Error stopping monitoring:', error);
        }
    }
}

// View management
async function toggleView(view) {
    const views = ['live', 'heatmap', 'analytics'];
    views.forEach(v => {
        document.getElementById(`${v}View`)?.classList.remove('active');
    });
    
    document.getElementById(`${view}View`)?.classList.add('active');
    
    // Toggle heatmap via API
    if (view === 'heatmap') {
        try {
            await fetch('/api/toggle_heatmap', { method: 'POST' });
        } catch (error) {
            console.error('Error toggling heatmap:', error);
        }
    }
    
    // Update video placeholder based on view
    const videoOverlay = document.querySelector('.video-overlay');
    const icons = {
        live: '📹',
        heatmap: '🗺️',
        analytics: '📊'
    };
    const texts = {
        live: 'Click Start Monitoring to begin',
        heatmap: 'Heatmap View',
        analytics: 'Analytics View'
    };
    
    if (videoOverlay) {
        videoOverlay.querySelector('.camera-icon').textContent = icons[view];
        videoOverlay.querySelector('.video-text').textContent = texts[view];
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new ZaytricsDashboard();
    
    // Add scroll animations
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, observerOptions);

    // Observe elements for scroll animations
    document.querySelectorAll('.feature-card, .dashboard-card').forEach(el => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(30px)';
        el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
        observer.observe(el);
    });
});

// Smooth scrolling for navigation
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});
// F1-themed enhancements
class F1ThemeEnhancements {
    constructor() {
        this.speedMetrics = {
            peopleCount: 0,
            processingSpeed: 0,
            accuracy: 99.9
        };
        this.init();
    }

    init() {
        this.addRacingSounds();
        this.enhanceMetrics();
        this.addSpeedEffect();
    }

    addSpeedEffect() {
        // Add speed lines effect to video container on data update
        const videoContainer = document.querySelector('.video-container');
        
        const observer = new MutationObserver((mutations) => {
            mutations.forEach((mutation) => {
                if (mutation.type === 'characterData' || mutation.target.classList.contains('stat-value')) {
                    this.triggerSpeedEffect(videoContainer);
                }
            });
        });

        const statElements = document.querySelectorAll('.stat-value');
        statElements.forEach(element => {
            observer.observe(element, {
                characterData: true,
                childList: true,
                subtree: true
            });
        });
    }

    triggerSpeedEffect(container) {
        // Create temporary speed lines
        for (let i = 0; i < 3; i++) {
            const line = document.createElement('div');
            line.style.cssText = `
                position: absolute;
                top: 0;
                left: ${Math.random() * 100}%;
                width: 2px;
                height: 100%;
                background: linear-gradient(to bottom, transparent, var(--primary), transparent);
                animation: speedLine 0.5s ease-out forwards;
                z-index: 10;
            `;
            
            container.appendChild(line);
            
            setTimeout(() => {
                line.remove();
            }, 500);
        }
    }

    enhanceMetrics() {
        // Add F1-style metric displays
        const metrics = document.querySelector('.hero-stats');
        if (metrics) {
            metrics.innerHTML += `
                <div class="stat">
                    <div class="stat-number">99.9%</div>
                    <div class="stat-label">Accuracy</div>
                </div>
            `;
        }
    }

    addRacingSounds() {
        // Optional: Add subtle racing sounds on interactions
        document.querySelector('.btn-primary').addEventListener('mouseenter', () => {
            // Could add subtle engine rev sound here
            console.log('Engine ready! 🏎️');
        });
    }
}

// Initialize F1 enhancements
document.addEventListener('DOMContentLoaded', () => {
    new ZaytricsDashboard();
    new F1ThemeEnhancements();
    
    // Add speed line animation to CSS
    const speedLineStyle = document.createElement('style');
    speedLineStyle.textContent = `
        @keyframes speedLine {
            0% {
                transform: translateY(-100%);
                opacity: 0;
            }
            50% {
                opacity: 0.8;
            }
            100% {
                transform: translateY(100%);
                opacity: 0;
            }
        }
    `;
    document.head.appendChild(speedLineStyle);
});